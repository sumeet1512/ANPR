# 🚗 Vehicle Speed Detection and Number Plate Recognition Using YOLO & OCR

This project detects and tracks vehicles in a video stream, estimates their speed, and recognizes number plates using Optical Character Recognition (OCR). The system is built using a custom-trained YOLOv8 model and PaddleOCR, and logs vehicle data into a CSV file.

![Preview](https://via.placeholder.com/800x400.png?text=Project+Demo+Preview) <!-- Replace with actual image/gif URL -->

## 📌 Features

- Vehicle detection and tracking with YOLOv8
- Real-time speed estimation based on pixel-to-meter calibration
- Number plate recognition using PaddleOCR
- Automatic data logging to CSV (date, time, speed, class, number plate)
- Annotated video output with object ID, class name, and speed

## 🛠️ Technologies Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- OpenCV
- NumPy
- Python 3.8+

## 📁 Directory Structure

├── best.pt # YOLOv8 trained model ├── inputt.mp4 # Input video file ├── numberplates_speedd.csv # Output CSV log file ├── speed_estimator.py # Main Python script ├── README.md # Project documentation


## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/vehicle-speed-ocr.git
   cd vehicle-speed-ocr

2. **Create a virtual environment**
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

Place your trained YOLO model (best.pt) and input video (input.mp4) in the project directory

**▶️ Usage**
Run the script to start processing:

python main.py
Note: Adjust the PIXELS_PER_METER value in the script according to your camera calibration for accurate speed estimation.


**🧪 Sample Output (CSV)**
date	time	track_id	class_name	speed_kmh	numberplate
2025-04-07	13:45:22	7	car	42.55	MH12AB1234


**🧠 Future Work**
Improve speed estimation using multi-frame smoothing or Kalman filter

Enhance OCR accuracy with focused cropping or deep OCR models

Add GUI using Streamlit or Gradio

Upload output video with annotations

**🤝 Contributing**
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
