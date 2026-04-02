#!/usr/bin/env python3
"""
Defines the Yolo class that uses the YOLOv3 algorithm for object detection.
Task 0: Initialize Yolo - loads the Darknet model and class names.
"""
import numpy as np
from tensorflow import keras


class Yolo:
    """
    Performs object detection using the YOLOv3 algorithm with a Darknet model.

    Public attributes:
        model       -- the loaded Darknet Keras model
        class_names -- list of class name strings
        class_t     -- box score threshold for initial filtering
        nms_t       -- IOU threshold for non-max suppression
        anchors     -- numpy array of anchor box dimensions
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo detector by loading the model and class names.

        Parameters:
            model_path   (str):            path to the Darknet Keras .h5 model
            classes_path (str):            path to the file of class names
            class_t      (float):          box score threshold for filtering
            nms_t        (float):         IOU threshold for non-max suppression
            anchors      (numpy.ndarray):  shape (outputs, anchor_boxes, 2)
                                           with [anchor_width, anchor_height]
        """
        # Load the pre-trained Darknet model from disk
        self.model = keras.models.load_model(model_path)

        # Read and strip class names from the classes file
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Store thresholds and anchor boxes as public attributes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
