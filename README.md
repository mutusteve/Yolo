# Real-Time Object Detection using YOLO and OpenCV

This project uses YOLO (You Only Look Once) with OpenCV to perform real-time object detection from a webcam.

## 📌 How It Works

1. **Load YOLO Model**  
   - The script loads the pre-trained YOLOv3 model (`yolov3.weights`) and its configuration file (`yolov3.cfg`).
   - It also loads class labels from `coco.names`, which contains names of 80 common objects.

2. **Initialize Webcam**  
   - The script captures live video using OpenCV’s `cv2.VideoCapture(0)`.
   - The frame size is set to **640x480** for optimized performance.

3. **Process Each Frame**  
   - Each frame is converted into a blob using `cv2.dnn.blobFromImage()`, which prepares the image for YOLO processing.
   - The blob is passed through the YOLO neural network to extract object detections.

4. **Extract Detections**  
   - YOLO provides **bounding boxes, class IDs, and confidence scores** for detected objects.
   - Confidence scores filter out low-probability detections.

5. **Apply Non-Maximum Suppression (NMS)**  
   - NMS removes overlapping bounding boxes and keeps the most accurate ones.
   - This helps reduce duplicate detections of the same object.

6. **Draw Bounding Boxes and Labels**  
   - The script draws rectangles around detected objects using OpenCV.
   - It labels each object with its class name and confidence percentage.

7. **Display Results**  
   - The annotated frame is displayed in a real-time window.
   - Press **'Q'** to exit the detection window.

---

## 🔧 Installation

1. Install dependencies:
   pip install opencv-python numpy

