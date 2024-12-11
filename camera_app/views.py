import cv2
import torch
import pytesseract
import re
from pathlib import Path
from datetime import datetime
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from PIL import Image
from yolov5.utils.plots import Annotator, colors
import base64
import numpy as np

# Homepage view
def homepage(request):
    return render(request, 'index.html')


# Initialize variables for drawing the rectangle
drawing = False
x1, y1, x2, y2 = -1, -1, -1, -1

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

stream_running = False
extracted_text = ""
result = {}

# Preprocessing function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Extract dates from text
def extract_dates(text):
    # Define date patterns
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',  # DD/MM/YYYY
        r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
        r'\b\d{2}/\d{2}/\d{2}\b',  # DD/MM/YY
        r'\b\d{2}-\d{2}-\d{2}\b',  # DD-MM-YY
        r'\b\d{2}/\d{4}\b',        # MM/YYYY
        r'\b\d{2}-\d{4}\b',        # MM-YYYY
    ]
    
    # Extract all matching dates
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return dates

# Classify expiration status
def classify_dates(dates):
    global result
    try:
        # Parse dates
        parsed_dates = []
        for date_text in dates:
            if '/' in date_text:
                if len(date_text.split('/')[-1]) == 2:  # Handle DD/MM/YY format
                    date_text = datetime.strptime(date_text, "%d/%m/%y").strftime("%d/%m/%Y")
                parsed_dates.append(datetime.strptime(date_text, "%d/%m/%Y"))
            elif '-' in date_text:
                if len(date_text.split('-')[-1]) == 2:  # Handle DD-MM-YY format
                    date_text = datetime.strptime(date_text, "%d-%m-%y").strftime("%d-%m/%Y")
                parsed_dates.append(datetime.strptime(date_text, "%d-%m-%Y"))

        if len(parsed_dates) == 0:
            return "No valid dates found."

        # Assign manufacturing and expiration dates
        if len(parsed_dates) == 1:
            expiry_date = parsed_dates[0]
            manuf_date = None
        else:
            manuf_date, expiry_date = sorted(parsed_dates)[:2]  # Assume the earlier date is manufacturing
        
        # Calculate days until expiry
        current_date = datetime.now()
        if expiry_date < current_date:
            status = "Expired"
            days_left = 0
        else:
            status = "Not Expired"
            days_left = (expiry_date - current_date).days
        
        result = {
            "Manufacturing Date": manuf_date.strftime("%d/%m/%Y") if manuf_date else "Not Available",
            "Expiration Date": expiry_date.strftime("%d/%m/%Y"),
            "Status": status,
            "Days Until Expiry": days_left
        }
        return result

    except Exception as e:
        return {"Error": str(e)}

# Camera feed generator
def generate_frames_expiry():
    global stream_running, extracted_text, result
    cap = cv2.VideoCapture(0)
    while stream_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess and extract text
        gray = preprocess_image(frame)
        extracted_text = pytesseract.image_to_string(gray)
        
        # Extract dates from the text
        extracted_dates = extract_dates(extracted_text)
        result = classify_dates(extracted_dates)

        # Display extracted text and dates
        cv2.putText(frame, f"Detected Text: {extracted_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Convert frame to JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

# Start camera stream
def start_stream(request):
    global stream_running
    stream_running = True
    return JsonResponse({"status": "stream started"})

# Stop camera stream
def stop_stream(request):
    global stream_running
    stream_running = False
    return JsonResponse({"status": "stream stopped"})

# Expiry detection page
def expiry_detection(request):
    global result
    return render(request, 'expiry_detection.html', {'task_title': 'Expiry Detection', 'task_heading': 'Expiry Detection', 'result': result})

# Video stream endpoint
def expiry_detection_stream(request):
    return StreamingHttpResponse(generate_frames_expiry(), content_type="multipart/x-mixed-replace; boundary=frame")

# Expiry detection page
def expiry_detection(request):
    return render(request, 'expiry_detection.html', {'task_title': 'Expiry Detection', 'task_heading': 'Expiry Detection'})





# Brand detection
stream_running = False
# Initialize the brand count dictionary
brand_counts = {
    'lays': 0,
    'oreo': 0,
    'nivea': 0,
}

# Function to generate frames for streaming
def generate_frames():
    global stream_running
    weights = Path(__file__).resolve().parent / "yolov5_brand/runs/train/exp7/weights/best.pt"
    data = Path(__file__).resolve().parent / "data.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO model
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(640, s=stride)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    while stream_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the brand detection (YOLO)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (imgsz, imgsz))  # Resize to model's input size
        img = torch.from_numpy(img).to(device).float()
        img = img.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] to [1, C, H, W]
        img /= 255.0  # Normalize

        # Run inference
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # Annotate detections
        annotator = Annotator(frame, line_width=3, example=str(names))
        detected_brands = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    detected_brands.append(names[int(cls)])

        # Update the brand counts
        for brand in detected_brands:
            if brand in brand_counts:
                brand_counts[brand] += 1


        # Convert frame to JPEG format
        frame = annotator.result()
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()

# Start video stream
def start_stream(request):
    global stream_running
    stream_running = True
    return JsonResponse({"status": "stream started"})

# Stop video stream
def stop_stream(request):
    global stream_running
    stream_running = False
    return JsonResponse({"status": "stream stopped"})


# Streaming view for the brand detection
def brand_detection_stream(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

# Get current brand counts
def get_brand_counts(request):
    return JsonResponse(brand_counts)

# Brand detection page
def brand_detection(request):
    return render(request, 'brand_detection.html', {'task_title': 'Brand Detection', 'task_heading': 'Brand Detection'})


# Item counting page
def item_counting(request):
    return render(request, 'item_counting.html', {'task_title': 'Item Counting', 'task_heading': 'Item Counting'})

# Global flag to control the stream
stream_running = False

# Initialize the fruit count dictionary
fruit_counts = {
    'apple': 0,
    'banana': 0,
    'orange': 0,
}

def generate_fruit_frames():
    global stream_running
    weights = Path(__file__).resolve().parent / "yolov5_fruit/runs/train/exp2/weights/best.pt"  # Update path for fruit detection
    data = Path(__file__).resolve().parent / "fruit_data.yaml"  # Update for fruit data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO model
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(640, s=stride)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    while stream_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (imgsz, imgsz))  # Resize to model's input size
        img = torch.from_numpy(img).to(device).float()
        img = img.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] to [1, C, H, W]
        img /= 255.0  # Normalize

        # Run inference
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        # Annotate detections
        annotator = Annotator(frame, line_width=3, example=str(names))
        detected_fruits = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                    detected_fruits.append(names[int(cls)])

        # Update the fruit counts
        for fruit in detected_fruits:
            if fruit in fruit_counts:
                fruit_counts[fruit] += 1

        # Convert frame to JPEG format
        frame = annotator.result()
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()

def start_fruit_stream(request):
    global stream_running
    stream_running = True
    return JsonResponse({"status": "stream started"})

def stop_fruit_stream(request):
    global stream_running
    stream_running = False
    return JsonResponse({"status": "stream stopped"})

def fruit_detection_stream(request):
    return StreamingHttpResponse(
        generate_fruit_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )

def get_fruit_counts(request):
    return JsonResponse(fruit_counts)


# Freshness detection page
def freshness_detection(request):
    return render(request, 'freshness_detection.html', {'task_title': 'Freshness Detection', 'task_heading': 'Freshness Detection'})
