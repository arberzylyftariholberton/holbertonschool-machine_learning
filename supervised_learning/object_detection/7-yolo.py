#!/usr/bin/env python3
"""
Defines the Yolo class that uses the YOLOv3 algorithm for object detection.
Task 7: Predict - orchestrates the full detection pipeline on a folder of
images and returns predictions paired with their file paths.
"""
import os
import cv2
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
                boxes           -- list of ndarrays
                                (grid_h, grid_w, anchors, 4)
                box_confidences -- list of ndarrays
                                (grid_h, grid_w, anchors, 1)
                box_class_probs -- list of ndarrays
                                (grid_h, grid_w, anchors, classes)
        """
        image_h, image_w = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, num_anchors)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, num_anchors)
            b_w = (anchor_w * np.exp(t_w)) / input_w
            b_h = (anchor_h * np.exp(t_h)) / input_h

            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters detected boxes by multiplying objectness confidence with class
        probabilities and discarding boxes below the class_t threshold.

        Parameters:
            boxes           (list of numpy.ndarray): boundary boxes per output
            box_confidences (list of numpy.ndarray):
                            objectness scores per output
            box_class_probs (list of numpy.ndarray):
                            class probabilities per output

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        all_boxes, all_classes, all_scores = [], [], []

        for box, confidence, class_prob in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * class_prob
            best_class = np.argmax(scores, axis=-1)
            best_score = np.max(scores, axis=-1)
            mask = best_score >= self.class_t
            all_boxes.append(box[mask])
            all_classes.append(best_class[mask])
            all_scores.append(best_score[mask])

        return (np.concatenate(all_boxes, axis=0),
                np.concatenate(all_classes, axis=0),
                np.concatenate(all_scores, axis=0))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-maximum suppression per class to remove highly overlapping
        boxes, retaining only the highest-scoring box when IOU >= nms_t.

        Parameters:
            filtered_boxes (numpy.ndarray): shape (N, 4)
            box_classes    (numpy.ndarray): shape (N,)
            box_scores     (numpy.ndarray): shape (N,)

        Returns:
            tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        result_boxes, result_classes, result_scores = [], [], []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

            order = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while len(cls_boxes) > 0:
                result_boxes.append(cls_boxes[0])
                result_classes.append(cls)
                result_scores.append(cls_scores[0])
                if len(cls_boxes) == 1:
                    break
                iou = self._iou(cls_boxes[0], cls_boxes[1:])
                survivors = np.where(iou < self.nms_t)[0] + 1
                cls_boxes = cls_boxes[survivors]
                cls_scores = cls_scores[survivors]

        return (np.array(result_boxes),
                np.array(result_classes),
                np.array(result_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a directory into memory using OpenCV.

        Parameters:
            folder_path (str): path to the directory containing image files

        Returns:
            tuple of (images, image_paths):
                images      (list of numpy.ndarray): loaded BGR images
                image_paths (list of str): corresponding full file paths
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            img = cv2.imread(full_path)
            if img is not None:
                images.append(img)
                image_paths.append(full_path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes images to the model's input
        dimensions using bicubic interpolation
        and normalises pixel values to the range [0, 1].

        Parameters:
            images (list of numpy.ndarray):
            original BGR images of arbitrary size

        Returns:
            tuple of (pimages, image_shapes):
                pimages      (numpy.ndarray): shape (ni, input_h, input_w, 3)
                image_shapes (numpy.ndarray): shape (ni, 2) — (height, width)
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            # Record original dimensions before any transformation
            image_shapes.append([img.shape[0], img.shape[1]])
            resized = cv2.resize(img, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Annotates an image with bounding boxes, class labels and scores,
        then displays it in an interactive window.

        Pressing 's' saves the annotated image to the 'detections/' directory.
        Any other key closes the window without saving.

        Parameters:
            image       (numpy.ndarray): the original unprocessed BGR image
            boxes       (numpy.ndarray): shape (N, 4) — (x1, y1, x2, y2)
            box_classes (numpy.ndarray): shape (N,) — class index per box
            box_scores  (numpy.ndarray): shape (N,) — confidence score per box
            file_name   (str): original file path; used as window title and
                               as the saved filename (basename only)
        """
        # Work on a copy to avoid mutating the original array
        annotated = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.class_names[box_classes[i]]
            score = round(float(box_scores[i]), 2)

            # Draw bounding box in blue with thickness 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)

            # Compose label: class name followed by score rounded to 2 d.p.
            label = "{} {:.2f}".format(class_name, score)

            # Render label in red, 5 pixels above the top-left corner
            cv2.putText(annotated, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA)

        # Window title is the bare filename (no directory)
        cv2.imshow(file_name, annotated)
        key = cv2.waitKey(0)

        if key == ord('s'):
            # Create the detections folder if it does not already exist
            os.makedirs('detections', exist_ok=True)
            save_path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(save_path, annotated)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Runs the full YOLOv3 detection pipeline on every image in a folder:
        load → preprocess → infer → decode → filter → NMS → display.

        Parameters:
            folder_path (str): path to the folder containing images to process

        Returns:
            tuple of (predictions, image_paths):
                predictions (list of tuple): each element is
                            (boxes, box_classes, box_scores) for one image
                image_paths (list of str): file paths matching each prediction
        """
        # Load all images then sort by path for deterministic ordering
        images, image_paths = self.load_images(folder_path)
        image_paths, images = zip(*sorted(zip(image_paths, images)))
        image_paths = list(image_paths)
        images = list(images)

        # Preprocess all images into a single batched array
        pimages, image_shapes = self.preprocess_images(images)

        # Run batched inference; result is a list of output-head arrays
        model_outputs = self.model.predict(pimages)

        predictions = []

        for idx, image in enumerate(images):
            # Collect each output head's slice for this image
            outputs = [model_outputs[o][idx]
                       for o in range(len(model_outputs))]

            # Decode raw predictions into boxes, confidences and class probs
            boxes, box_confs, box_probs = self.process_outputs(
                outputs, image_shapes[idx])

            # Discard low-confidence detections
            boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confs, box_probs)

            # Suppress redundant overlapping boxes per class
            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes, box_classes, box_scores)

            # Display using the bare filename as the window title
            file_name = os.path.basename(image_paths[idx])
            self.show_boxes(image, boxes, box_classes, box_scores, file_name)

            predictions.append((boxes, box_classes, box_scores))

        return predictions, image_paths

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
        ix1 = np.maximum(box[0], boxes[:, 0])
        iy1 = np.maximum(box[1], boxes[:, 1])
        ix2 = np.minimum(box[2], boxes[:, 2])
        iy2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area_box + area_boxes - inter)
