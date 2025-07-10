# 🍅 Tomato Ripeness Computer Vision

![Tomato Ripeness Detection](/images/tomato.png)

A computer vision system built using **Python** and **YOLOv8m** to detect and classify tomato ripeness levels from **photos**, **videos**, and **real-time camera input**. This project uses the **TomatOD** dataset to train a model that distinguishes between unripe, semi-ripe, and ripe tomatoes.

## 🚀 Features

- Detects tomatoes in various input sources: images, video files, and live webcam.
- Classifies tomato ripeness into multiple stages.
- Based on **YOLOv8m** for real-time object detection.
- Easily customizable and modular Python-based system.

## 🧠 Tech Stack

- Python
- YOLOv8m (Ultralytics)
- OpenCV
- Jupyter / script-based architecture

## 📊 Dataset

We used the [**TomatOD dataset**](https://www.kaggle.com/datasets/nexuswho/tomatod), a tomato object detection dataset containing over 4,000 annotated images from greenhouse environments.

📦 **TomatOD Dataset (Kaggle Version)**  
🔗 https://www.kaggle.com/datasets/nexuswho/tomatod

📝 **Original Project**:  
GitHub - https://github.com/up2metric/tomatOD  
Published by: [Up2Metric](https://up2metric.com)

The dataset provides annotation for tomato bounding boxes and ripeness stages (unripe, semi-ripe, ripe) with high-quality YOLO-formatted labels.

## 📁 Project Structure

.
├── images/ # Project assets (e.g., tomato.png)
├── models/ # YOLOv8m trained model weights
├── scripts/
│ ├── detect_image.py # Detect tomatoes from image
│ ├── detect_video.py # Detect tomatoes from video
│ └── detect_cam.py # Real-time detection using webcam
├── tomatod/ # Preprocessed dataset (optional)
├── requirements.txt # Project dependencies
└── README.md # This file

bash
Copy
Edit

## 📄 Documentation

Setup, training, and usage instructions are documented in full here:

📘 [Project Documentation](https://docs.google.com/document/d/1v_Ew9WaYm2xAltw7PYFq-moW7TPsmuWcNacvYpDnWVg/edit?usp=sharing)

## 🧪 Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/hwasyui/tomatoripeness.git
cd tomatoripeness
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run detection

Image

bash
Copy
Edit
python scripts/detect_image.py --source path/to/image.jpg
Video

bash
Copy
Edit
python scripts/detect_video.py --source path/to/video.mp4
Camera

bash
Copy
Edit
python scripts/detect_cam.py
⚠️ Ensure your YOLOv8m model weights are in the models/ folder or update the script with the correct path.

👥 Team Members
Name	GitHub / Handle
Angelica Suti Whiharto	@hwasyui
Intan Kumala Pasya	@tannpsy
Rason Yudha Pati N	@RasonYudha4
Yoel Heardly Sirait	@heardlyyoel

📌 Repository
🔗 GitHub: https://github.com/hwasyui/tomatoripeness

yaml
Copy
Edit

---

Let me know if you want to add:

- A training guide for custom YOLOv8 weights  
- A performance/mAP evaluation section  
- Example outputs (e.g., image with bounding boxes)

I'm happy to tailor this further!
