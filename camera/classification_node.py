#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import ultralytics



class ObjectDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO('')  # Load your custom .pt model
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Perform object detection
        results = self.model(cv_image)

        # Display results
        cv2.imshow("Object Detection", results.render()[0])
        cv2.waitKey(1)

def main():
    rospy.init_node('object_detection_node', anonymous=True)
    ObjectDetectionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
