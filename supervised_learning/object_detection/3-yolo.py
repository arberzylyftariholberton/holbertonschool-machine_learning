#!/usr/bin/env python3
"""
Defines the Yolo class that uses the YOLOv3 algorithm for object detection.
Task 3: Non-max Suppression - removes highly overlapping boxes per class,
keeping only the highest-scoring box when IOU exceeds nms_t.
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

    def process_outputs(self, outputs, image_size):
        """
        Decodes raw Darknet model outputs into boundary boxes, confidences,
        and class probabilities scaled to the original image dimensions.

        Parameters:
            outputs    (list of numpy.ndarray): raw predictions, each of shape
                       (grid_h, grid_w, anchor_boxes, 4 + 1 + classes)
            image_size (numpy.ndarray): [image_height, image_width] of the
                       original (unprocessed) image

        Returns:
            tuple of (boxes, box_confidences, box_class_probs):
                boxes       -- list of ndarrays (grid_h, grid_w, anchors, 4)
                               with (x1, y1, x2, y2) in original-image pixels
                box_confidences -- list of ndarrays
                                    (grid_h, grid_w, anchors, 1)
                box_class_probs -- list of ndarrays
                                    (grid_h, grid_w, anchors, classes)
        """
        image_h, image_w = image_size

        # Retrieve the model's expected input dimensions
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            # Extract raw prediction components
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            # Build grid offsets: c_x varies along columns, c_y along rows
            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)

            # Decode centre coordinates: sigmoid + grid offset, normalised
            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            # Decode dimensions: anchor * exp(t), normalised by input size
            anchor_w = self.anchors[i, :, 0].reshape(1, 1, num_anchors)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, num_anchors)
            b_w = (anchor_w * np.exp(t_w)) / input_w
            b_h = (anchor_h * np.exp(t_h)) / input_h

            # Convert centre format to corner format in original-image pixels
            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            # Stack into (grid_h, grid_w, anchor_boxes, 4)
            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            # Sigmoid objectness confidence, keep trailing dim → (..., 1)
            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            # Sigmoid class probabilities → (..., classes)
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters detected boxes by multiplying objectness confidence with class
        probabilities and discarding boxes below the class_t threshold.

        Parameters:
            boxes           (list of numpy.ndarray): boundary boxes per output,
                            each shape (grid_h, grid_w, anchor_boxes, 4)
            box_confidences (list of numpy.ndarray):
                            objectness scores per output,
                            each shape (grid_h, grid_w, anchor_boxes, 1)
            box_class_probs (list of numpy.ndarray):
                            class probabilities per output,
                            each shape (grid_h, grid_w, anchor_boxes, classes)

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes (numpy.ndarray): shape (N, 4)
                box_classes (numpy.ndarray): shape (N,) predicted class index
                box_scores  (numpy.ndarray): shape (N,) confidence * class prob
        """
        all_boxes = []
        all_classes = []
        all_scores = []

        for box, confidence, class_prob in zip(
                boxes, box_confidences, box_class_probs):

            # Box score = objectness confidence × per-class probability
            scores = confidence * class_prob

            # Predicted class is the one with the highest combined score
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)

            # Boolean mask: keep only boxes above the threshold
            mask = best_score >= self.class_t

            all_boxes.append(box[mask])
            all_classes.append(best_class[mask])
            all_scores.append(best_score[mask])

        # Concatenate results from all output scales
        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-maximum suppression per class to remove highly overlapping
        boxes, retaining only the highest-scoring box when IOU >= nms_t.

        Parameters:
            filtered_boxes (numpy.ndarray): shape (N, 4) — (x1, y1, x2, y2)
            box_classes    (numpy.ndarray): shape (N,) — class index per box
            box_scores     (numpy.ndarray): shape (N,) — score per box

        Returns:
            tuple of (box_predictions, predicted_box_classes,
                        predicted_box_scores)
            ordered by class then by descending score within each class.
        """
        result_boxes = []
        result_classes = []
        result_scores = []

        # Run NMS independently for each unique predicted class
        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            # Sort by descending score so the best box is always first
            order = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            # Greedy NMS loop: keep top box, remove overlapping candidates
            while len(cls_boxes) > 0:
                result_boxes.append(cls_boxes[0])
                result_classes.append(cls)
                result_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                # Compute IOU of the kept box against all remaining boxes
                iou = self._iou(cls_boxes[0], cls_boxes[1:])

                # Retain only boxes with IOU strictly below the threshold
                survivors = np.where(iou < self.nms_t)[0] + 1
                cls_boxes = cls_boxes[survivors]
                cls_scores = cls_scores[survivors]

        # Stack all surviving boxes across classes
        box_predictions = np.array(result_boxes)
        predicted_box_classes = np.array(result_classes)
        predicted_box_scores = np.array(result_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def _iou(box, boxes):
        """
        Computes Intersection-over-Union between one reference box and N boxes.

        Parameters:
            box   (numpy.ndarray): shape (4,) — (x1, y1, x2, y2)
            boxes (numpy.ndarray): shape (N, 4)

        Returns:
            numpy.ndarray: shape (N,) — IOU values in [0, 1]
        """
        # Coordinates of the intersection rectangle
        ix1 = np.maximum(box[0], boxes[:, 0])
        iy1 = np.maximum(box[1], boxes[:, 1])
        ix2 = np.minimum(box[2], boxes[:, 2])
        iy2 = np.minimum(box[3], boxes[:, 3])

        # Intersection area (zero when boxes do not overlap)
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)

        # Individual areas and union
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - inter

        return inter / union
