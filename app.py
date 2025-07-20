from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import time
import winsound

app = Flask(__name__)

# Load the pre-trained mask detection model
model = tf.keras.models.load_model('mask_detection_model.h5')

# Load Haar cascades for frontal and profile face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Alert sound file
alert_file = "siren-alert-96052.mp3"

# Cooldown timer setup
alert_cooldown = 3  # seconds between alerts
last_alert_time = 0

# Function to play alert sound (non-blocking)
def play_alert():
    winsound.PlaySound(alert_file, winsound.SND_FILENAME | winsound.SND_ASYNC)

# Function to predict mask or no mask
def predict_mask(face_img, model):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.equalizeHist(face_img)
    face_img = cv2.resize(face_img, (128, 128))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    prediction = model.predict(face_img, verbose=0)
    return prediction[0][0]

# Initialize the video camera
camera = cv2.VideoCapture(0)

def generate_frames():
    global last_alert_time
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect frontal faces first
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # If none, try profile faces
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Apply 10% inward cropping to reduce background
            crop_margin_x = int(0.1 * w)
            crop_margin_y = int(0.1 * h)
            x1 = max(x + crop_margin_x, 0)
            y1 = max(y + crop_margin_y, 0)
            x2 = min(x + w - crop_margin_x, frame.shape[1])
            y2 = min(y + h - crop_margin_y, frame.shape[0])
            face_img = frame[y1:y2, x1:x2]

            confidence = predict_mask(face_img, model)
            label = "Mask" if confidence < 0.6 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Play alert if "No Mask" and cooldown time has passed
            if label == "No Mask":
                current_time = time.time()
                if current_time - last_alert_time > alert_cooldown:
                    play_alert()
                    last_alert_time = current_time

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Concatenate frame one by one and show result
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Homepage route
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
