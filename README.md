

# 😷 Face Mask Detection with Live Alert System

## 🔍 Overview

This project is a real-time face mask detection system that uses deep learning and computer vision to determine whether individuals are wearing a mask. The system captures live video from a webcam, detects faces using Haar Cascade classifiers, and classifies each face as either "Mask" or "No Mask" using a trained Convolutional Neural Network (CNN). If a person is detected without a mask, the system can trigger alerts such as sounds, on-screen warnings, or event logging.

---

## 🧰 Technologies Used

* **Python**
* **TensorFlow/Keras** – For model training and prediction
* **OpenCV** – For face detection and video stream handling
* **NumPy & Matplotlib** – For data handling and visualization
* **scikit-learn** – For data preprocessing and evaluation
* **Flask** *(optional)* – For deploying as a web application
* **playsound/pygame** *(optional)* – For audible alerts

---

## 📊 Dataset

* Source: Public datasets (e.g., Kaggle) of images of people with and without masks
* Preprocessing steps include:

  * Resizing images to a fixed size (e.g., 100x100 or 128x128)
  * Grayscale conversion (for face detection)
  * Normalization of pixel values (0 to 1)

---

## 🧠 Model Training

* A CNN model is designed using Keras with layers such as Conv2D, MaxPooling2D, Flatten, and Dense
* The model is trained to perform **binary classification**: Mask vs. No Mask
* After training, the model is saved as `mask_detector_model.h5`

---

## 📸 Real-Time Detection Logic

* Live video is captured using `cv2.VideoCapture()`
* Haar Cascades detect face regions in each frame
* Each detected face is cropped, preprocessed, and passed through the trained model
* Predictions are displayed in real-time with labels and bounding boxes
* If “No Mask” is detected, an alert is triggered (sound, color box, or log)

---

## 🔔 Alert System

* Play a beep sound for unmasked faces
* Show a red warning box on the screen
* Optionally log the detection with a timestamp

---

## 🌐 Optional Flask Deployment

* The detection system can be deployed via a Flask web application
* Provides routes for a homepage and real-time video stream
* Enables remote monitoring via a web browser

---

## ✅ Project Deliverables

* Trained model: `mask_detector_model.h5` (Wasn't able to includ ein the repository due to file restrictions)
 
* Real-time detection script: `realtime_detection.py`
* Flask app (optional): `app.py`
* Setup instructions: `README.md`
* Demo video of detection system in action
* GitHub repository with all code and assets

---

## 📌 Setup Instructions

1. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the real-time detection script:

   ```bash
   python realtime_detection.py
   ```
3. (Optional) Launch Flask web app:

   ```bash
   python app.py
   ```

   Then open your browser at `http://localhost:5000`

---
