from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import yaml

app = Flask(__name__)

def load_user_mapping():
    if os.path.exists('user_mapping.yaml'):
        with open('user_mapping.yaml', 'r') as file:
            user_id_mapping = yaml.load(file, Loader=yaml.FullLoader)
        return user_id_mapping
    else:
        return {}

def save_user_mapping(user_id_mapping):
    with open('user_mapping.yaml', 'w') as file:
        yaml.dump(user_id_mapping, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/read_faces', methods=['POST'])
def read_faces():
    if not os.path.exists('images'):
        os.makedirs('images')

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Change Path Here
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    count = 0

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Change Path Here

    name = request.form.get('user_name')
    user_id_mapping = load_user_mapping()

    if name not in user_id_mapping:
        user_id_mapping[name] = len(user_id_mapping) + 1

    user_id = user_id_mapping[name]
    save_user_mapping(user_id_mapping)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            cv2.imwrite(f"./images/Users.{user_id}.{count}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k < 30 or count >= 50:
            break

    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()

    # Trigger the training step after reading faces
    train_faces()

    return jsonify({'message': 'Faces read successfully', 'user_id_mapping': user_id_mapping})

@app.route('/train_faces', methods=['POST'])
def train_faces():
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        return faceSamples, ids

    print("\n[INFO] Training faces...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    return jsonify({'message': f'{len(np.unique(ids))} faces trained. Exiting Program'})

@app.route('/recognize_faces', methods=['GET'])
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0
    names = ['None'] + list(load_user_mapping().keys())
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

            if 0 <= id < len(names):  # Ensure 'id' is a valid index
                if confidence < 70:
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "Who are you?"
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
    return jsonify({'message': 'Recognize Faces'})
if __name__ == "__main__":
    app.run(debug=True)
