# Smart Surveillance System

An AI-powered real-time smart surveillance system using Python, YOLO and OpenCV.

## Features

- Real-time camera monitoring
- Knife detection
- Appliance detection
- Person detection
- Multi-model YOLO detection
- Real-time alerts
- Local video processing

## Technologies

- Python
- YOLO
- Ultralytics
- OpenCV
- Roboflow

## Project Structure

```text
Smart-Surveillance/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── detect.py
│
├── scripts/
│   └── download.py
│
├── models/
│   ├── best.pt
│   ├── appliance.pt
│   └── yolov8n.pt
│
└── screenshots/



**Installation**

#Install the required packages:

pip install -r requirements.txt
Run

#Run the surveillance system:

python src/detect.py
