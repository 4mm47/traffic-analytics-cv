# Traffic Analytics - Computer Vision Pipeline

A real-time traffic analytics system that uses **YOLOv8** to detect and count vehicles from traffic camera footage.

## What It Does

- Detects **cars, trucks, buses, motorcycles, and pedestrians** in real-time
- Displays a **live count** of each vehicle type currently on screen
- Processes any traffic camera video feed

## Screenshots

![Detection Example 1](image%201.PNG)

![Detection Example 2](image%202.PNG)

## Tech Stack

- **YOLOv8** (Ultralytics) — Object detection
- **OpenCV** — Video processing and display
- **Python 3.14**

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/traffic-analytics-cv.git
cd traffic-analytics-cv

# Create a virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Place your traffic video in the project folder as `test_video.mp4`
2. Run the script:

```bash
python main.py
```

3. Press **q** or click the **X** button to close the video window.

## How It Works

1. Each video frame is fed into a pre-trained YOLOv8 model
2. The model returns bounding boxes with class labels and confidence scores
3. Detected objects are filtered by class (car, truck, bus, motorcycle, person)
4. A live counter is displayed on the video in real-time

## Future Improvements

- [ ] Object tracking with unique IDs (ByteTrack / DeepSORT)
- [ ] Line-crossing counter for directional traffic flow
- [ ] CSV data logging with timestamps
- [ ] Speed estimation using homography
- [ ] Live RTSP camera stream support
