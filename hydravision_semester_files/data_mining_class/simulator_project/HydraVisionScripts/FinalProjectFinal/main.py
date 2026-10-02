#!/usr/bin/env python3
"""
TurtleBot4 Object Detection Benchmark Runner
=============================================
Orchestrates benchmarking of YOLOv8, Faster R-CNN, and SSD models
under controlled conditions in the Gazebo simulator.

Usage:
    ros2 run <your_pkg> main.py
    OR
    python3 main.py --config config.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import time
import psutil
import os
import json
import argparse
import threading
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

# --- Model imports (install as needed) ---
# pip install ultralytics torchvision torch
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — YOLOv8 will be skipped.")

try:
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        ssd300_vgg16,
        SSD300_VGG16_Weights,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch/torchvision not installed — Faster R-CNN and SSD will be skipped.")


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class Detection:
    bbox: List[float]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    inference_time_ms: float
    detections: List[Detection]
    cpu_percent: float
    ram_mb: float


@dataclass
class ModelMetrics:
    model_name: str
    total_frames: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    inference_times_ms: List[float] = field(default_factory=list)
    cpu_samples: List[float] = field(default_factory=list)
    ram_samples_mb: List[float] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)

    # ── Computed properties ──
    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.inference_times_ms, 95)) if self.inference_times_ms else 0.0

    @property
    def avg_fps(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
        duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
        return (len(self.frame_timestamps) - 1) / duration if duration > 0 else 0.0

    @property
    def avg_cpu_percent(self) -> float:
        return float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0

    @property
    def avg_ram_mb(self) -> float:
        return float(np.mean(self.ram_samples_mb)) if self.ram_samples_mb else 0.0

    def summary(self) -> Dict:
        return {
            "model": self.model_name,
            "frames_processed": self.total_frames,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "avg_fps": round(self.avg_fps, 2),
            "avg_cpu_percent": round(self.avg_cpu_percent, 2),
            "avg_ram_mb": round(self.avg_ram_mb, 2),
        }


# ──────────────────────────────────────────────
# Ground-truth provider
# ──────────────────────────────────────────────

class GroundTruthProvider:
    """
    Loads ground-truth annotations for precision/recall calculation.

    Supports two modes:
      1. JSON file  — {frame_id: [{bbox, class_id}, ...]}
      2. Gazebo model states topic — subscribe to /gazebo/model_states
         and project 3-D poses to 2-D image coords using camera intrinsics.

    For quick testing, a synthetic GT generator is included.
    """

    def __init__(self, gt_path: Optional[str] = None,
                 image_width: int = 640, image_height: int = 480):
        self.gt_path = gt_path
        self.annotations: Dict[int, List[Dict]] = {}
        self.image_width = image_width
        self.image_height = image_height
        self._load()

    def _load(self):
        if self.gt_path and Path(self.gt_path).exists():
            with open(self.gt_path) as f:
                raw = json.load(f)
            self.annotations = {int(k): v for k, v in raw.items()}
            print(f"[GT] Loaded {len(self.annotations)} annotated frames from {self.gt_path}")
        else:
            print("[GT] No annotation file found — using synthetic ground truth.")

    def get(self, frame_id: int) -> List[Dict]:
        """Return list of {bbox:[x1,y1,x2,y2], class_id:int} for frame_id."""
        if frame_id in self.annotations:
            return self.annotations[frame_id]
        # Synthetic: one centred object per frame
        cx, cy = self.image_width // 2, self.image_height // 2
        w, h = self.image_width // 4, self.image_height // 4
        return [{"bbox": [cx - w, cy - h, cx + w, cy + h], "class_id": 0}]


def iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    gt_boxes: List[Dict],
    det_boxes: List[Detection],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int]:
    """Returns (TP, FP, FN) for one frame."""
    matched_gt = set()
    tp = fp = 0
    for det in det_boxes:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            score = iou(gt["bbox"], det.bbox)
            if score > best_iou:
                best_iou, best_idx = score, i
        if best_iou >= iou_thresh and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ──────────────────────────────────────────────
# Model wrappers
# ──────────────────────────────────────────────

class BaseDetector:
    name: str = "base"

    def infer(self, frame: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    def warmup(self, frame: np.ndarray, n: int = 3):
        for _ in range(n):
            self.infer(frame)


class YOLOv8Detector(BaseDetector):
    name = "YOLOv8"

    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_thresh: float = 0.4, device: str = "cpu"):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is not installed.")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.device = device
        print(f"[YOLOv8] Loaded model from {model_path} on {device}")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.conf_thresh,
                             device=self.device, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                    class_name=self.model.names[int(box.cls[0])],
                ))
        return detections


class FasterRCNNDetector(BaseDetector):
    name = "FasterRCNN"

    def __init__(self, conf_thresh: float = 0.5, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/torchvision is not installed.")
        self.device = torch.device(device)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.eval().to(self.device)
        self.transforms = weights.transforms()
        self.labels = weights.meta["categories"]
        self.conf_thresh = conf_thresh
        print(f"[FasterRCNN] Loaded pretrained model on {device}")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        inp = self.transforms(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inp)[0]
        detections = []
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=float(score),
                class_id=int(label),
                class_name=self.labels[int(label)],
            ))
        return detections


class SSDDetector(BaseDetector):
    name = "SSD"

    def __init__(self, conf_thresh: float = 0.4, device: str = "cpu"):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/torchvision is not installed.")
        self.device = torch.device(device)
        weights = SSD300_VGG16_Weights.DEFAULT
        self.model = ssd300_vgg16(weights=weights)
        self.model.eval().to(self.device)
        self.transforms = weights.transforms()
        self.labels = weights.meta["categories"]
        self.conf_thresh = conf_thresh
        print(f"[SSD] Loaded pretrained model on {device}")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        inp = self.transforms(tensor).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(inp)[0]
        detections = []
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score < self.conf_thresh:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=float(score),
                class_id=int(label),
                class_name=self.labels[int(label)],
            ))
        return detections


# ──────────────────────────────────────────────
# ROS2 benchmark node
# ──────────────────────────────────────────────

class BenchmarkNode(Node):
    """
    Subscribes to the TurtleBot4 camera topic, runs each model on
    every frame, and accumulates metrics.
    """

    def __init__(self, config: Dict):
        super().__init__("detection_benchmark_node")
        self.config = config
        self.bridge = CvBridge()

        # Build model list
        self.detectors: List[BaseDetector] = self._build_detectors()
        if not self.detectors:
            raise RuntimeError("No detectors could be initialised. Check dependencies.")

        self.metrics: Dict[str, ModelMetrics] = {
            d.name: ModelMetrics(model_name=d.name) for d in self.detectors
        }

        self.gt_provider = GroundTruthProvider(
            gt_path=config.get("ground_truth_path"),
            image_width=config.get("image_width", 640),
            image_height=config.get("image_height", 480),
        )

        self.frame_count = 0
        self.max_frames = config.get("max_frames", 300)
        self.iou_thresh = config.get("iou_threshold", 0.5)
        self.output_dir = Path(config.get("output_dir", "benchmark_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_annotated = config.get("save_annotated_frames", False)
        self._shutdown_requested = False
        self._process = psutil.Process(os.getpid())
        self._done = False
        self._lock = threading.Lock()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        camera_topic = config.get("camera_topic", "/oakd/rgb/preview/image_raw")
        self.sub = self.create_subscription(
            Image, camera_topic, self._image_callback, qos
        )
        self.get_logger().info(
            f"Subscribed to {camera_topic}. "
            f"Will process {self.max_frames} frames per model run."
        )

        # Warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for d in self.detectors:
            self.get_logger().info(f"Warming up {d.name}...")
            d.warmup(dummy)

    # ── detector factory ──────────────────────

    def _build_detectors(self) -> List[BaseDetector]:
        detectors = []
        cfg = self.config
        device = cfg.get("device", "cpu")
        enabled = cfg.get("models", ["yolov8", "fasterrcnn", "ssd"])

        if "yolov8" in enabled and YOLO_AVAILABLE:
            try:
                detectors.append(YOLOv8Detector(
                    model_path=cfg.get("yolov8_weights", "yolov8n.pt"),
                    conf_thresh=cfg.get("conf_threshold", 0.4),
                    device=device,
                ))
            except Exception as e:
                print(f"[WARN] Could not load YOLOv8: {e}")

        if "fasterrcnn" in enabled and TORCH_AVAILABLE:
            try:
                detectors.append(FasterRCNNDetector(
                    conf_thresh=cfg.get("conf_threshold", 0.5),
                    device=device,
                ))
            except Exception as e:
                print(f"[WARN] Could not load FasterRCNN: {e}")

        if "ssd" in enabled and TORCH_AVAILABLE:
            try:
                detectors.append(SSDDetector(
                    conf_thresh=cfg.get("conf_threshold", 0.4),
                    device=device,
                ))
            except Exception as e:
                print(f"[WARN] Could not load SSD: {e}")

        return detectors

    # ── ROS callback ──────────────────────────

    def _image_callback(self, msg: Image):
        if self._done:
            return

        if self.frame_count >= self.max_frames:
            self._finalise()
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        frame_id = self.frame_count
        gt_boxes = self.gt_provider.get(frame_id)

        for detector in self.detectors:
            m = self.metrics[detector.name]
            cpu_before = self._process.cpu_percent(interval=None)
            t0 = time.perf_counter()
            try:
                detections = detector.infer(frame)
            except Exception as e:
                self.get_logger().error(f"[{detector.name}] Inference error: {e}")
                continue
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            cpu_after = self._process.cpu_percent(interval=None)
            ram_mb = self._process.memory_info().rss / 1024 / 1024
            tp, fp, fn = match_detections(gt_boxes, detections, self.iou_thresh)
            m.total_frames += 1
            m.true_positives += tp
            m.false_positives += fp
            m.false_negatives += fn
            m.inference_times_ms.append(latency_ms)
            m.cpu_samples.append((cpu_before + cpu_after) / 2.0)
            m.ram_samples_mb.append(ram_mb)
            m.frame_timestamps.append(t1)
            if self.save_annotated:
                self._save_frame(frame.copy(), detections, gt_boxes,
                                 detector.name, frame_id)

        self.frame_count += 1
        if self.frame_count % 50 == 0:
            self.get_logger().info(
                f"Processed {self.frame_count}/{self.max_frames} frames"
            )

    # ── frame visualisation ───────────────────

    def _save_frame(self, frame, detections, gt_boxes, model_name, frame_id):
        for gt in gt_boxes:
            x1, y1, x2, y2 = [int(v) for v in gt["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "GT", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                        (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        out_path = self.output_dir / "frames" / model_name
        out_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path / f"frame_{frame_id:05d}.jpg"), frame)

    # ── finalisation ──────────────────────────

    def _finalise(self):
        if self._done:
            return

        
        print("FINALISE: step 1 — computing summaries", flush=True)
        summaries = [m.summary() for m in self.metrics.values()]

        print("FINALISE: step 2 — printing table", flush=True)
        _print_table(summaries)

        print("FINALISE: step 3 — saving JSON", flush=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = self.output_dir / f"results_{ts}.json"
        with open(out_json, "w") as f:
            json.dump({"timestamp": ts, "config": self.config,
                    "results": summaries}, f, indent=2)
        self.get_logger().info(f"Results saved to {out_json}")

        print("FINALISE: step 4 — importing Metrics", flush=True)
        try:
            from metrics import MetricsReporter
            print("FINALISE: step 5 — running reporter", flush=True)
            reporter = MetricsReporter(summaries, str(self.output_dir), ts)
            print("FINALISE: step 6 — saving CSV", flush=True)
            reporter.save_csv()
            print("FINALISE: step 7 — saving plots", flush=True)
            reporter.save_plots()
            print("FINALISE: step 8 — plots done", flush=True)
        except Exception as e:
            print(f"FINALISE: reporter failed with {e} — skipping", flush=True)

        print("FINALISE: step 9 — setting shutdown flag", flush=True)
        self._shutdown_requested = True
        print("FINALISE: done", flush=True)


# ──────────────────────────────────────────────
# Console table
# ──────────────────────────────────────────────

def _print_table(summaries: List[Dict]):
    cols = ["model", "precision", "recall", "f1_score",
            "avg_latency_ms", "p95_latency_ms", "avg_fps",
            "avg_cpu_percent", "avg_ram_mb", "frames_processed"]
    widths = [14] + [16] * (len(cols) - 1)
    header = "".join(str(c).ljust(w) for c, w in zip(cols, widths))
    sep = "-" * len(header)
    print("\n" + sep)
    print("  BENCHMARK RESULTS")
    print(sep)
    print(header)
    print(sep)
    for s in summaries:
        row = "".join(str(s.get(c, "")).ljust(w) for c, w in zip(cols, widths))
        print(row)
    print(sep + "\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def load_config(path: Optional[str]) -> Dict:
    defaults = {
        "camera_topic": "/oakd/rgb/preview/image_raw",
        "models": ["yolov8", "fasterrcnn", "ssd"],
        "yolov8_weights": "yolov8n.pt",
        "conf_threshold": 0.4,
        "iou_threshold": 0.5,
        "device": "cpu",
        "max_frames": 100,
        "image_width": 640,
        "image_height": 480,
        "ground_truth_path": None,
        "output_dir": "benchmark_results",
        "save_annotated_frames": False,
    }
    if path and Path(path).exists():
        with open(path) as f:
            user = yaml.safe_load(f)
        defaults.update(user)
    return defaults


def main():
    parser = argparse.ArgumentParser(description="TurtleBot4 Detection Benchmark")
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    rclpy.init()
    node = BenchmarkNode(config)
    try:
        while rclpy.ok() and not node._shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()