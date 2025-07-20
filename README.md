Face Mask Detection with Live Alert System
Objective: Detect if people are wearing face masks in real-time using a webcam.
Tools: Python, OpenCV, TensorFlow/Keras, Flask, Haar Cascades
Mini-Guide:
Collect or use Kaggle dataset of masked/unmasked faces
Preprocess images (resize, grayscale)
Train CNN model using Keras
Integrate model with OpenCV video stream
Use Haar Cascades for face detection
Add logic to alert when no mask is detected
Deploy with Flask (optional)
Deliverables: Trained model, real-time detection script, short video demo, GitHub repo 
📁 Step 1: Setup Environment
Libraries to install:

tensorflow

keras

opencv-python

numpy

matplotlib (for visualization)

scikit-learn (for splitting data and evaluation)

flask (for web app deployment)

Optionally install:

jupyterlab or notebook for experimentation

📦 Step 2: Dataset Preparation
Sources:

Download a dataset of masked and unmasked faces from Kaggle.

Data format:

Images of people with and without masks.

If annotations are in .xml, they could be in Pascal VOC format or Haar Cascade classifiers (e.g., haarcascade_frontalface_default.xml).

Preprocessing:

Resize images to a fixed size (e.g., 100x100 or 128x128).

Convert to grayscale if needed (for Haar).

Normalize pixel values (scale to 0-1).

🧠 Step 3: Model Training
Use:

Keras with TensorFlow backend to define a CNN model.

Steps:

Define CNN model architecture (Conv2D → MaxPooling → Dense).

Use binary classification: Masked vs. Unmasked.

Train the model using model.fit() on preprocessed images.

Save model using model.save('mask_detector_model.h5').

📸 Step 4: Face Detection with OpenCV
Use Haar Cascade XML file:

Example: haarcascade_frontalface_default.xml

Steps:

Load the cascade using cv2.CascadeClassifier().

Detect faces in webcam stream frames using .detectMultiScale().

🔍 Step 5: Real-Time Detection Logic
Steps:

Capture video stream using cv2.VideoCapture().

For each frame:

Detect face(s) using Haar.

Crop face region.

Resize and preprocess for CNN.

Predict using trained model.

Display results (label: “Mask” or “No Mask”).

Trigger alert if “No Mask”.

🔔 Step 6: Add Alert System
Options:

Play a beep sound using playsound or pygame.

Show red warning box on screen.

Log events with timestamp.

🌐 Step 7: Optional Flask Deployment
Create Flask app with:

Route for homepage

Route for video stream (use Response object with cv2.imencode)

Embed webcam feed in HTML via <img src="/video_feed">

Allow remote access via browser

Results:-
mask_detector_model.h5: Trained model file.

realtime_detection.py: Script for live webcam detection and alert.

app.py: Flask deployment script.

static/ and templates/: Flask folder structure.

requirements.txt: All libraries used.

README.md: Instructions and setup.

Short video demo of detection in action.

GitHub Repo: Upload everything there.

#Setup Instructions
Run the app.py script to start the Flask app.
Access the app in your browser at http://localhost:5000. or whatever address is shown
#The Face mask detection system is a computer vision application that uses deep learning models to detect whether a person is wearing a face mask or not. The system uses a webcam to capture video frames and applies a pre-trained deep learning model to classify each frame as either a masked or unmasked face will run a face mask detection system.
The system uses a pre-trained deep learning model to classify each frame as either a masked or unmasked face. The model is trained on a dataset of images of people wearing and not wearing masks.