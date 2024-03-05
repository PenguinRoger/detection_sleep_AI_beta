from flask import Flask, render_template, jsonify, Response, request
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import threading
from multiprocessing import Queue
import threading

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config["TEMPLATES_AUTO_RELOAD"] = True

mixer.init()
sound = mixer.Sound('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/work/alarm.wav')

face = cv2.CascadeClassifier('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/haar_cascade_files/haarcascade_righteye_2splits.xml')

model = load_model('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/work/models_detext.h5')
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = None    
lpred = None
cap = None
cap_opened = False

# สร้าง Queue เพื่อส่งคะแนนระหว่างกระบวนการ
score_queue = Queue()

thread = None

terminate_event = threading.Event()

def detect_drowsiness():
    global score, count, rpred, lpred, thicc, cap, cap_opened
    while not terminate_event.is_set():  
        if cap_opened:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Unable to read frame from camera.")
                continue

            height, width = frame.shape[:2]
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray is None:
                print("Unable to convert frame to gray scale.")
                continue
            
            faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
            left_eye = leye.detectMultiScale(gray, minSize=(30, 30))
            right_eye = reye.detectMultiScale(gray, minSize=(30, 30))
            
            if len(faces) == 0 or (len(left_eye) == 0 and len(right_eye) == 0):
                cv2.putText(frame, "No face or eyes found", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                continue
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y+h, x:x+w]
                count += 1
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(24, 24, -1)
                r_eye = np.expand_dims(r_eye, axis=0)
                predict_x = model.predict(r_eye)
                rpred = np.argmax(predict_x, axis=1)
                if rpred[0] == 1:
                    lbl = 'Open'
                if rpred[0] == 0:
                    lbl = 'Closed'
                break

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y+h, x:x+w]
                count += 1
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(24, 24, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                predict_x = model.predict(l_eye)
                lpred = np.argmax(predict_x, axis=1)
                if lpred[0] == 1:
                    lbl = 'Open'
                if lpred[0] == 0:
                    lbl = 'Closed'
                break
            
            if rpred is not None and lpred is not None:
                if rpred[0] == 0 and lpred[0] == 0:
                    score += 1
                    score_queue.put(score)  
                    cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    score -= 1
                    score_queue.put(score)  
                    cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if score < 0:
                score = 0
            
            cv2.putText(frame, 'Score:'+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
            if score > 15:
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                except Exception as e:
                    print("Error playing sound:", str(e))
                if thicc < 16:
                    thicc += 2
                else:
                    thicc -= 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
            
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

    print("Camera released.")

    release_camera()

def release_camera():
    global cap, cap_opened
    if cap_opened:
        cap.release()
        print("Camera released.")
        cap_opened = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blank.html')
def new_page():
    return render_template('blank.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global thread, cap, cap_opened, terminate_event
    if not thread or not thread.is_alive():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap_opened = True  
        thread = threading.Thread(target=detect_drowsiness)
        thread.start()
        return jsonify({'status': 'Detection started'})
    else:
        return jsonify({'status': 'Detection already started'})


@app.route('/stop_detection', methods=['POST']) 
def stop_detection():
    global thread, cap, cap_opened, terminate_event
    if thread and thread.is_alive():
        terminate_event.set()
        release_camera()  
        thread.join()  
        return jsonify({'status': 'Detection stopped'})
    else:
        return jsonify({'status': 'Detection not running'})


if __name__ == '__main__':
    app.run(debug=True)