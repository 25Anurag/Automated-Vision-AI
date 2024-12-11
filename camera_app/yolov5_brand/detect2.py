import cv2
import torch
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors

def run_as_stream(weights='runs/train/exp7/weights/best.pt', imgsz=640, conf_thres=0.25, iou_thres=0.45):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img = cv2.resize(frame, (imgsz, imgsz))
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # Normalize
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # Add batch dimension

        # Run inference
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Annotate detections
        annotator = Annotator(frame, line_width=3, example=str(names))
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Convert frame to JPEG format
        frame = annotator.result()
        _, buffer = cv2.imencode('.jpg', frame)
        yield buffer.tobytes()

    cap.release()
    cv2.destroyAllWindows()
