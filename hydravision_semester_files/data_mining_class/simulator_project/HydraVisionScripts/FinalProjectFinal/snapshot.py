# snapshot.py — saves one camera frame to frame.jpg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class Snapshot(Node):
    def __init__(self):
        super().__init__("snapshot")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, "/oakd/rgb/preview/image_raw", self._cb, 10)

    def _cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite("frame.jpg", frame)
        print("Saved frame.jpg", flush=True)
        rclpy.shutdown()

rclpy.init()
node = Snapshot()
rclpy.spin(node)