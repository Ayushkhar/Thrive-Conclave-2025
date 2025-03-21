import os
import cv2
import pygame
import cvzone
import math
from flask import Flask, render_template, Response
from ultralytics import YOLO

pygame.mixer.init()
sirene_sound = pygame.mixer.Sound('sirene.mp3')
app = Flask(__name__)

model = YOLO('fire.pt')
classnames = ['fire']
fire_detected = False
frame_count = 0
frame_skip = 2
fire_intensity = 0

cap = cv2.VideoCapture('fire2.mp4')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to generate frames from video and detect fire
def generate_frames():
    global fire_intensity, fire_detected, frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        result = model(frame, stream=True)

        fire_intensity = 0
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    color = (0, 255, 0)  # Default Green (Unused, just a placeholder)
                    if confidence <= 70:
                        color = (0, 255, 255)  # Yellow (Low Intensity)
                    elif confidence <= 90:
                        color = (0, 165, 255)  # Orange (Medium Intensity)
                    else:
                        color = (0, 0, 255)  # Red (High Intensity)
                        
                    fire_intensity = max(fire_intensity, confidence)  # Track max intensity
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                    cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 30],
                                       scale=1.5, thickness=2, offset=10)
                    
                    if not fire_detected:
                        sirene_sound.play(loops=-1)
                        fire_detected = True

        # Fire Intensity Indicator Bar
        cv2.rectangle(frame, (20, 450), (620, 470), (50, 50, 50), -1)  # Background Bar
        cv2.rectangle(frame, (20, 450), (20 + int(fire_intensity * 6), 470), (0, 0, 255), -1)  # Intensity Bar
        cv2.putText(frame, f'Fire Intensity: {fire_intensity}%', (25, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Send the frame to the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
