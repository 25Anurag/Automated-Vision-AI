Here's the updated README with the reference website link added:

---

# Automated Vision AI

Automated Vision AI is a cutting-edge solution that leverages computer vision and machine learning techniques to detect brands, expiry dates, and fruit types in real-time via a webcam feed. Built with Python, Django, YOLOv5, and Tesseract, it allows seamless integration of brand detection, expiry date extraction, and freshness recognition into your web application.

## Features

- **Brand Detection**: Identify popular brands like Lays, Oreo, and Nivea from a live video stream using YOLOv5-based object detection.
- **Expiry Date Extraction**: Capture expiration dates from product images using OCR (Tesseract) and classify their status (Expired or Not Expired).
- **Fruit Detection**: Recognize different fruits in real-time and count occurrences based on a pre-trained YOLOv5 model.
- **Stream Control**: Start and stop live video streams with ease.
- **Real-time Analytics**: Track and display counts of detected brands, expiry statuses, and fruits.

## Requirements

- Python 3.x
- Django
- OpenCV
- YOLOv5
- Tesseract-OCR
- PyTorch
- Other dependencies (listed in `requirements.txt`)

## Installation

### 1. Clone the repository

git clone https://github.com/yourusername/Automated-Vision-AI.git
cd Automated-Vision-AI


### 2. Install dependencies

pip install -r requirements.txt


### 3. Install Tesseract OCR

- **Windows**: Download and install Tesseract from [here](https://github.com/UB-Mannheim/tesseract/wiki) and update the `pytesseract.pytesseract.tesseract_cmd` path in the code.
- **Linux**: Run `sudo apt install tesseract-ocr` to install Tesseract.

### 4. Configure YOLOv5 Models

Ensure the YOLOv5 models (for brand, fruit, etc.) are properly trained and placed in the `yolov5_brand` and `yolov5_fruit` directories.

### 5. Run Django Development Server

```bash
python manage.py runserver
```

Your application should now be accessible at `http://127.0.0.1:8000`.

## Usage

- **Homepage**: Access the main page from the browser to view all available options.
- **Brand Detection**: Navigate to the `Brand Detection` page to view the live webcam stream and detect brands.
- **Expiry Detection**: Navigate to the `Expiry Detection` page for OCR-based expiry date detection.
- **Fruit Detection**: Visit the `Fruit Detection` page to detect and count fruits in real-time.

### Start Video Stream

To start a live stream:

POST /brand-detection/start/
POST /expiry-detection/start/
POST /fruit-detection/start/

### Stop Video Stream

To stop a live stream:

POST /brand-detection/stop/
POST /expiry-detection/stop/
POST /fruit-detection/stop/


### Get Detection Counts

You can check the current count of detected brands or fruits:

GET /get-brand-counts/
GET /get-fruit-counts/


## Reference Website

You can access a live demo of Automated Vision AI at the following link:

[Automated Vision AI Demo](https://889cjwxm-8000.inc1.devtunnels.ms/)
