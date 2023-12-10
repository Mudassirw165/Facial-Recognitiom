from flask import Flask, render_template, request
from flask_socketio import SocketIO
import cv2
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('read_faces')
def read_faces(data):
    user_id = data['user_id']
    user_name = data['user_name']

    if not os.path.exists('images'):
        os.makedirs('images')

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Change Path Here
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    count = 0

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the images directory
            cv2.imwrite(f"./images/Users.{user_id}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)
        # Press Escape to end the program.
        k = cv2.waitKey(100) & 0xff
        if k < 30 or count >= 50:
            break

    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()

    socketio.emit('read_faces_response', {'message': 'Read Faces'})

@socketio.on('train_faces')
def train_faces():
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Change Path Here

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n[INFO] Training faces...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    socketio.emit('train_faces_response', {'message': 'Train Faces'})

@socketio.on('recognize_faces')
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    face_cascade_Path = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(face_cascade_Path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    names = ['None', 'Aleem', 'Usama', 'Azhar']  # add a name into this list
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(confidence))
            else:
                id = "Unknown"
                confidence = "  {0}%".format(round(confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()

    socketio.emit('recognize_faces_response', {'message': 'Recognize Faces'})

if __name__ == '__main__':
    app.run(port=int(os.environ.get('PORT', 5000)))

