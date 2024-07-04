import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import base64
from gtts import gTTS
from playsound import playsound
import numpy as np
from food_facts import food_facts

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model = YOLO("best.pt")  # Load your YOLOv8 model

stop_video = False  # Flag to stop video capture

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    detected_items = []
    for result in results:
        for box in result.boxes:
            detected_items.append(result.names[int(box.cls[0])])
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, detected_items

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame = base64.b64encode(buffer).decode('utf-8')
    return frame

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('start_video')
def handle_start_video():
    global stop_video
    stop_video = False  # Reset the flag
    cap = cv2.VideoCapture(0)
    while True:
        if stop_video:
            break
        success, img = cap.read()
        if not success:
            break
        result_img, detected_items = predict_and_detect(model, img, classes=[], conf=0.5)
        frame = encode_frame(result_img)
        emit('video_frame', {'frame': frame, 'detected_items': detected_items})
        socketio.sleep(0.1)  # Control the frame rate
    cap.release()

@socketio.on('stop_video')
def handle_stop_video():
    global stop_video
    stop_video = True

if __name__ == '__main__':
    socketio.run(app, debug=True)
