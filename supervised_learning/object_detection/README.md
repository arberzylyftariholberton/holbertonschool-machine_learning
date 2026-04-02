# Object Detection with YOLOv3

A from-scratch implementation of the **YOLOv3 (You Only Look Once v3)** object detection pipeline built on top of a pre-trained Darknet Keras model. The project covers the full inference pipeline: loading a model, decoding raw predictions, filtering detections, applying Non-Maximum Suppression, preprocessing images, and visualizing results with annotated bounding boxes.

---

## Table of Contents

- [Background](#background)
- [How YOLOv3 Works](#how-yolov3-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Tasks Breakdown](#tasks-breakdown)
- [Key Concepts](#key-concepts)
- [Model Architecture](#model-architecture)
- [Dataset — COCO](#dataset--coco)
- [Results](#results)
- [Limitations](#limitations)
- [Author](#author)

---

## Background

Object detection is one of the most fundamental and challenging tasks in computer vision. Unlike image classification, which simply assigns a label to an entire image, object detection requires the model to simultaneously:

1. **Locate** every object of interest in the image (bounding box regression)
2. **Classify** each detected object into one of the known categories

Traditional approaches relied on sliding windows or region proposal networks, which were accurate but too slow for real-time use. **YOLO** changed that by reframing detection as a single regression problem — the entire image is passed through the network exactly once, and the output encodes all bounding boxes and class probabilities simultaneously. This makes YOLO significantly faster than two-stage detectors like Faster R-CNN while remaining competitive in accuracy.

**YOLOv3**, introduced by Joseph Redmon and Ali Farhadi in 2018, is the third major iteration of the YOLO family. It introduces multi-scale predictions across three different grid resolutions, a deeper feature extractor called Darknet-53, and the use of independent logistic classifiers instead of softmax — making it better at detecting multiple overlapping objects and objects that belong to more than one category (e.g. "person" and "athlete").

This project implements the complete **post-processing pipeline** for a pre-trained YOLOv3 Darknet model from scratch using Python, NumPy, TensorFlow/Keras, and OpenCV — without relying on any high-level detection libraries.

---

## How YOLOv3 Works

### 1. Single-pass detection

The input image is resized to a fixed resolution (typically 416×416 pixels) and passed through the Darknet-53 convolutional backbone. Unlike two-stage detectors, YOLOv3 makes all its predictions in a single forward pass through the network.

### 2. Multi-scale predictions

YOLOv3 produces three output tensors from three different layers of the network, each operating at a different spatial resolution:

| Output | Grid Size | Best for |
|--------|-----------|----------|
| Output 1 | 13 × 13 | Large objects |
| Output 2 | 26 × 26 | Medium objects |
| Output 3 | 52 × 52 | Small objects |

Each grid cell in every output predicts **3 bounding boxes** (one per anchor), giving a total of `(13×13 + 26×26 + 52×52) × 3 = 10,647` candidate boxes per image before any filtering.

### 3. Anchor boxes

Rather than predicting absolute box dimensions, YOLOv3 predicts offsets relative to pre-defined **anchor boxes**. The 9 anchors (3 per scale) are determined by running k-means clustering on the bounding box dimensions in the COCO training set. This gives the model a strong prior about typical object shapes and sizes, making training faster and more stable.

### 4. Bounding box decoding

For each cell at position `(c_x, c_y)` in the grid, the model outputs four raw values `(t_x, t_y, t_w, t_h)`. These are decoded into actual box coordinates using:

```
b_x = sigmoid(t_x) + c_x   (normalised by grid width)
b_y = sigmoid(t_y) + c_y   (normalised by grid height)
b_w = p_w × exp(t_w)       (normalised by model input width)
b_h = p_h × exp(t_h)       (normalised by model input height)
```

Where `p_w` and `p_h` are the anchor width and height. The sigmoid on `t_x` and `t_y` constrains the predicted centre to stay within the responsible grid cell, which prevents duplicate predictions from neighbouring cells and stabilises training.

The normalised centre coordinates `(b_x, b_y)` and dimensions `(b_w, b_h)` are then converted to corner coordinates `(x1, y1, x2, y2)` and scaled to the original image dimensions.

### 5. Objectness and class scores

Each predicted box also outputs:
- An **objectness confidence** score (sigmoid of raw value): the probability that an object of any class is present inside the box
- **Class probabilities** (sigmoid of raw values, one per class): independent binary classifiers for each of the 80 COCO classes

The final **box score** for each class is computed as:
```
box_score = objectness_confidence × class_probability
```

### 6. Filtering and NMS

Since 10,647 candidate boxes are generated per image, a two-step filtering process is applied:
1. **Score threshold filtering**: discard all boxes with `box_score < class_t`
2. **Non-Maximum Suppression (NMS)**: for each class, suppress overlapping boxes with `IoU ≥ nms_t`, keeping only the highest-scoring one

---

## Project Structure

```
object_detection/
│
├── 0-yolo.py          # Task 0: Yolo class initialization
├── 1-yolo.py          # Task 1: Process raw model outputs into boxes
├── 2-yolo.py          # Task 2: Filter boxes by class score threshold
├── 3-yolo.py          # Task 3: Non-Maximum Suppression per class
├── 4-yolo.py          # Task 4: Load images from a folder
├── 5-yolo.py          # Task 5: Resize and normalise images
├── 6-yolo.py          # Task 6: Draw and display annotated detections
├── 7-yolo.py          # Task 7: End-to-end prediction pipeline
│
├── yolo.h5            # Pre-trained Darknet Keras model (not included)
├── coco_classes.txt   # List of 80 COCO class names
│
├── yolo_images/
│   └── yolo/          # Folder of test images for detection
│
└── detections/        # Output folder for saved annotated images (auto-created)
```

Each numbered file is **cumulative** — every file builds on the previous one by adding a new method to the `Yolo` class, following the pattern required by the project specification. This means `7-yolo.py` is the most complete version and includes all methods from all prior tasks.

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Runtime |
| TensorFlow / Keras | 2.x | Model loading and inference |
| NumPy | 1.x | Array operations and mathematical decoding |
| OpenCV (`cv2`) | 4.x | Image I/O, resizing, and visualization |

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/arberzylyftari123/holbertonschool-machine_learning.git
cd holbertonschool-machine_learning/supervised_learning/object_detection
```

**2. Install Python dependencies**

```bash
pip install tensorflow numpy opencv-python
```

**3. Download the model and class file**

Place the following files in the `object_detection/` directory:
- `yolo.h5` — pre-trained YOLOv3 Darknet model in Keras `.h5` format
- `coco_classes.txt` — list of 80 COCO class names, one per line

**4. Add test images**

Place your images inside `yolo_images/yolo/`. Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`.

---

## Usage

Each task file is standalone and can be tested with its corresponding main script.

```bash
# Task 0 — Initialize and inspect the Yolo class
python3 0-main.py

# Task 1 — Decode raw model outputs into bounding boxes
python3 1-main.py

# Task 2 — Filter detections by score threshold
python3 2-main.py

# Task 3 — Apply Non-Maximum Suppression
python3 3-main.py

# Task 4 — Load images from disk
python3 4-main.py

# Task 5 — Preprocess images for the model
python3 5-main.py

# Task 6 — Display annotated detections (interactive window)
python3 6-main.py

# Task 7 — Run the full end-to-end detection pipeline
python3 7-main.py
```

### Interactive image window (Tasks 6 and 7)

When a detection window opens for each image, you have two options:
- Press **`s`** — saves the annotated image to the `detections/` folder (created automatically if it does not exist) and closes the window
- Press **any other key** — closes the window without saving

---

## Tasks Breakdown

### Task 0 — Initialize Yolo (`0-yolo.py`)

Sets up the `Yolo` class by:
- Loading the Darknet Keras model from `model_path` using `tf.keras.models.load_model`
- Reading class names line by line from `classes_path`, stripping whitespace
- Storing `class_t`, `nms_t`, and `anchors` as public instance attributes

**Public attributes:**

| Attribute | Type | Description |
|---|---|---|
| `model` | Keras Model | The loaded Darknet network |
| `class_names` | list of str | The 80 COCO class name strings |
| `class_t` | float | Box score threshold for initial filtering |
| `nms_t` | float | IoU threshold for Non-Maximum Suppression |
| `anchors` | ndarray `(outputs, anchor_boxes, 2)` | Anchor box width and height per scale |

---

### Task 1 — Process Outputs (`1-yolo.py`)

Adds `process_outputs(self, outputs, image_size)`.

Iterates over each of the three output tensors from the model and applies the full YOLOv3 decoding pipeline:
- Uses `np.meshgrid` to create a grid of cell offsets `(c_x, c_y)` matching the current output's grid dimensions
- Applies sigmoid to `t_x` and `t_y`, adds cell offsets, and normalises by grid size to get fractional centre coordinates
- Multiplies anchor dimensions by `exp(t_w)` and `exp(t_h)`, normalises by the model's input size
- Converts from centre format `(b_x, b_y, b_w, b_h)` to corner format `(x1, y1, x2, y2)` and scales to original image pixels
- Applies sigmoid independently to objectness confidence (keeping the trailing dimension of 1) and to all class probabilities

**Returns:** a tuple of three lists — `(boxes, box_confidences, box_class_probs)` — one entry per output scale.

---

### Task 2 — Filter Boxes (`2-yolo.py`)

Adds `filter_boxes(self, boxes, box_confidences, box_class_probs)`.

For each output scale independently:
- Computes the element-wise product `box_score = confidence × class_probability` across all classes
- Uses `argmax` to identify the most likely class for each box and `max` to get its score
- Applies a boolean mask retaining only boxes where `max_score >= self.class_t`
- Accumulates surviving boxes, class indices, and scores using `np.concatenate`

This step reduces candidate boxes from ~10,647 down to only the most confident detections before NMS.

**Returns:** `(filtered_boxes, box_classes, box_scores)` as flat 1D/2D arrays.

---

### Task 3 — Non-Maximum Suppression (`3-yolo.py`)

Adds `non_max_suppression(self, filtered_boxes, box_classes, box_scores)` and the helper `iou(self, box1, box2)`.

**IoU helper** computes:
```
Intersection = max(0, min(x2) - max(x1)) × max(0, min(y2) - max(y1))
Union = Area1 + Area2 - Intersection
IoU = Intersection / Union
```

**Greedy NMS** per unique class:
1. Sort all boxes for the class by descending confidence score
2. Select the highest-scoring box and add it to the output
3. Compute IoU between the selected box and all remaining candidates
4. Remove all candidates with `IoU > self.nms_t`
5. Repeat until no candidates remain for this class
6. Move to the next class

The output is sorted by class first, then by descending score within each class.

**Returns:** `(box_predictions, predicted_box_classes, predicted_box_scores)`.

---

### Task 4 — Load Images (`4-yolo.py`)

Adds the static method `load_images(folder_path)`.

Scans the given directory for image files with extensions `.jpg`, `.jpeg`, `.png`, or `.bmp`. For each file:
- Constructs the full path with `os.path.join`
- Attempts to read the file with `cv2.imread`
- Skips any file that OpenCV cannot decode (returns `None`)
- Appends valid images and their paths to the output lists

Implemented as a `@staticmethod` because it does not depend on any instance state — it only interacts with the filesystem.

**Returns:** `(images, image_paths)` — a list of NumPy BGR image arrays and their corresponding absolute file paths.

---

### Task 5 — Preprocess Images (`5-yolo.py`)

Adds `preprocess_images(self, images)`.

Prepares a list of raw images for batched model inference:
- Reads the model's required input `height` and `width` from `self.model.input.shape`
- Records each image's original `(height, width)` in `image_shapes` for later use when rescaling predictions back to original image coordinates
- Resizes each image to `(input_h, input_w)` using **bicubic interpolation** (`cv2.INTER_CUBIC`), which produces smoother results than bilinear or nearest-neighbour at the cost of slightly more compute
- Normalises pixel values from the range `[0, 255]` to `[0, 1]` by dividing by `255.0`
- Stacks all preprocessed images into a single NumPy array for batched inference

**Returns:**
- `pimages` — ndarray of shape `(ni, input_h, input_w, 3)`, ready to pass to `model.predict()`
- `image_shapes` — ndarray of shape `(ni, 2)` containing original `[height, width]` per image

---

### Task 6 — Show Boxes (`6-yolo.py`)

Adds `show_boxes(self, image, boxes, box_classes, box_scores, file_name)`.

Draws detection results on a copy of the original image using OpenCV drawing primitives:
- Iterates over every predicted box and converts its coordinates to integers
- Looks up the class name from `self.class_names` using the predicted class index
- Draws a **blue rectangle** (`(255, 0, 0)` in BGR) around each detected object with line thickness 2
- Formats a label string as `"class_name score"` with the score rounded to 2 decimal places
- Renders the label in **red** (`(0, 0, 255)`) using `cv2.putText` with `FONT_HERSHEY_SIMPLEX` (scale 0.5, thickness 1, `LINE_AA` for anti-aliasing), positioned 5 pixels above the top-left corner of each box
- Displays the annotated image in a named OpenCV window
- On keypress `s`: creates the `detections/` directory if needed and saves the annotated image using `cv2.imwrite`

---

### Task 7 — Predict (`7-yolo.py`)

Adds `predict(self, folder_path)`.

Ties the entire pipeline together into a single public method:

```
load_images()
    → sort by path (deterministic ordering)
        → preprocess_images()
            → model.predict()         ← single batched forward pass
                → process_outputs()   ← per image, per scale
                    → filter_boxes()
                        → non_max_suppression()
                            → show_boxes()
                                → collect predictions
```

Key implementation notes:
- Images are sorted alphabetically by path before processing to ensure consistent output order regardless of filesystem ordering
- `model.predict()` is called once on the entire batch, then per-image slices are extracted for post-processing
- The window title passed to `show_boxes` is the **bare filename** only (via `os.path.basename`), not the full path
- Each image's predictions are appended as a `(boxes, box_classes, box_scores)` tuple

**Returns:** `(predictions, image_paths)`

---

## Key Concepts

### Anchor Boxes
Pre-defined bounding box shapes derived from k-means clustering on COCO ground truth annotations. YOLOv3 uses 9 anchors in total — 3 per output scale — ranging from small anchors at the 52×52 scale for detecting tiny objects (e.g. `[10, 13]`, `[16, 30]`, `[33, 23]`) to large anchors at the 13×13 scale for detecting large objects (e.g. `[116, 90]`, `[156, 198]`, `[373, 326]`). The model predicts offsets and scale factors relative to these anchors rather than raw absolute coordinates, which makes optimisation significantly easier.

### Grid Cells and Responsible Cell
The image is divided into a grid at each scale. Each grid cell is "responsible" for predicting objects whose centres fall within that cell's region. The sigmoid constraint on `t_x` and `t_y` ensures the predicted centre cannot move outside its assigned cell, preventing two adjacent cells from making duplicate predictions for the same object.

### Sigmoid vs Softmax for Classes
Unlike YOLOv1 and v2 which used softmax for class predictions (forcing exactly one class per box and making classes mutually exclusive), YOLOv3 uses independent sigmoid activations for each class. This allows a single detection to belong to multiple overlapping categories simultaneously — important for hierarchical or overlapping labels such as "dog" and "animal", or "person" and "athlete".

### IoU (Intersection over Union)
The standard metric for measuring spatial overlap between two axis-aligned bounding boxes:
```
IoU = Area of Intersection / Area of Union
```
Ranges from 0.0 (no overlap) to 1.0 (perfect overlap). Used in NMS to suppress redundant detections, and during training as a matching criterion between predictions and ground truth annotations.

### Non-Maximum Suppression (NMS)
Because multiple grid cells and anchor types can simultaneously predict the same physical object, NMS is essential for removing duplicate detections. Greedy NMS iteratively selects the highest-scoring box, then suppresses all remaining boxes that overlap it beyond the `nms_t` threshold. It is applied per class so that detections from different classes are never incorrectly suppressed against each other.

### Score Threshold Filtering
Before NMS, all boxes with a combined `confidence × class_probability` score below `class_t` are discarded outright. This removes low-confidence background noise early in the pipeline and drastically reduces the number of boxes that need to be processed by the more expensive NMS step — from potentially thousands to typically just tens or hundreds.

### Bicubic Interpolation
Used during image resizing to the model's fixed input dimensions. Bicubic interpolation uses a weighted average over a 4×4 pixel neighbourhood, producing smoother and visually higher-quality resized images compared to bilinear (2×2) or nearest-neighbour. This is particularly valuable for preserving fine-grained texture and edge information that convolutional layers depend on for accurate detection.

### Objectness Score
Separate from class probabilities, the objectness score measures how likely it is that any object at all is present inside a given box. High objectness but low class probability suggests the model sees something but cannot identify it confidently. Multiplying the two together for the final box score ensures that both conditions — "something is here" and "I know what it is" — must be satisfied simultaneously for a box to survive filtering.

---

## Model Architecture

The pre-trained model (`yolo.h5`) is a **Darknet-53** based YOLOv3 architecture loaded into Keras. Key properties:

| Property | Value |
|---|---|
| Input shape | `(None, 416, 416, 3)` |
| Number of output heads | 3 (one per detection scale) |
| Output shapes | `(None, 13, 13, 3, 85)`, `(None, 26, 26, 3, 85)`, `(None, 52, 52, 3, 85)` |
| Total classes | 80 (COCO) |
| Trainable parameters | ~62 million |
| Model size on disk | ~236 MB |

The `85` values in each output tensor break down as:
- `4` — bounding box regression (`t_x`, `t_y`, `t_w`, `t_h`)
- `1` — objectness confidence
- `80` — per-class probabilities (one sigmoid per COCO class)

---

## Dataset — COCO

The model is trained on the [Microsoft COCO (Common Objects in Context)](https://cocodataset.org/) dataset, one of the largest and most widely used benchmarks in object detection, segmentation, and captioning.

**80 detectable classes include:**

| Category | Examples |
|---|---|
| People | person |
| Vehicles | bicycle, car, motorbike, bus, truck, boat, aeroplane, train |
| Animals | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| Food | banana, apple, sandwich, orange, pizza, donut, cake, hot dog |
| Electronics | laptop, mouse, keyboard, cell phone, tv monitor, microwave, oven |
| Household | chair, sofa, bed, toilet, sink, refrigerator, clock, vase |
| Sports | frisbee, skis, snowboard, sports ball, kite, skateboard, surfboard, tennis racket |
| Accessories | backpack, umbrella, handbag, tie, suitcase |

---

## Results

When run on sample images from the YOLO test set, the pipeline correctly identifies and localises common objects with annotated class labels and confidence scores displayed above each bounding box. Typical outputs include:

- Dogs, cats, and horses correctly labelled with species and high confidence
- Vehicles (cars, trucks, buses) localised on street scenes with tight bounding boxes
- People detected individually even in moderately crowded scenes
- Multiple object categories detected simultaneously in the same image

All annotated output images are saved to the `detections/` folder when the `s` key is pressed during display, allowing easy review of results after running the pipeline.

---

## Limitations

- **Fixed input size**: images are resized to 416×416 regardless of their original aspect ratio, which can introduce distortion in very wide or very tall inputs and may reduce accuracy on such images.
- **No aspect-ratio-preserving resize (letterboxing)**: a production implementation would pad images with grey borders to maintain aspect ratio before resizing, which typically improves detection quality.
- **Static anchors**: the 9 anchor boxes are fixed to the COCO dataset distribution and may not be optimal for specialised domains with unusually shaped objects (e.g. satellite/aerial imagery, medical imaging).
- **CPU post-processing**: the NMS and decoding steps run in pure NumPy on the CPU. For production-grade inference speed, these would typically be implemented as TensorFlow operations or offloaded to a dedicated inference runtime like TensorRT or ONNX Runtime.
- **No tracking**: each call to `predict` processes images independently with no temporal information, so there is no object identity or tracking across video frames.

---

## Author

**Arber Zylyftari**  
Machine Learning Student — Holberton School  
[GitHub](https://github.com/arberzylyftari123) · [Medium](https://medium.com/@arberzylyftari123)