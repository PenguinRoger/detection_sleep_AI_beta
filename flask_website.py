from flask import Flask, render_template, Response, request
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

app = Flask(__name__)

mixer.init()
sound = mixer.Sound('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/alarm.wav')

face = cv2.CascadeClassifier('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/haar cascade files/haarcascade_righteye_2splits.xml')


lbl = ['Close', 'Open']
model = load_model('C:/Users/NathanRoger/Desktop/DevCode/HTML/AI/Drowsiness_detection/models_detext.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

def detect():
    global rpred, lpred, score, thicc

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]
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
            l_eye = frame[y:y + h, x:x + w]
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

        if rpred[0] == 0 and lpred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            score -= 1
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score > 15:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            try:
                sound.play()
            except:
                pass
            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():
    global cap  # เพิ่มบรรทัดนี้เพื่ออ้างถึงตัวแปร cap ที่ประกาศเป็น global

    if 'action' in request.form:
        if request.form['action'] == 'start':
            cap.release()  # ปล่อยทรัพยากรของกล้องเมื่อเริ่มต้นการทำงาน
            cap = cv2.VideoCapture(0)  # เปิดการเชื่อมต่อกับกล้อง
        elif request.form['action'] == 'stop':
            cap.release()  # ปล่อยทรัพยากรของกล้องเมื่อหยุดการทำงาน
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
