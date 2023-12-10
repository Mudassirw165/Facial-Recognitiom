import cv2
import numpy as np
from PIL import Image
import os







# Check if folder exists
def read_faces():
    if not os.path.exists('images'):
        os.makedirs('images')

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Change Path Here
    cam = cv2.VideoCapture(0)
    cam.set(3,640)
    cam.set(4,480)
    count = 0

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Change Path Here
    # For each person, enter one unique numeric face id
    face_id = input('\n enter user id (MUST be an integer) and press <return> -->  ')
    name = input('\n please enter your name and press enter --> ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")


    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            # Save the captured image into the images directory
            cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        # Press Escape to end the program.
        k = cv2.waitKey(100) & 0xff
        if k < 30:
            break
        # Take 30 face samples and stop video. You may increase or decrease the number of
        # images. The more the better while training the model.
        elif count >= 50:
             break

    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()



def train_faces():
    #Directory path name where the face images are stored.
    path = './images/'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #Haar cascade file
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Change Path Here

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            # convert it to grayscale
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n[INFO] Training faces...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into the current directory.
    recognizer.write('trainer.yml')
    print("\n[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))




def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    face_cascade_Path = 'haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(face_cascade_Path)

    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0
    # names related to ids: The names associated with the ids: 1 for Mohamed, 2 for Jack, etc...
    names = ['None', 'Aleem', 'Usama', 'Azhar']  # add a name into this list
    # Video Capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    # Min Height and Width for the window size to be recognized as a face
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
                # Unknown Face
                id = "Who are you ?"
                confidence = "  {0}%".format(round(confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)
        # Escape to exit the webcam / program
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print("\n [INFO] Exiting Program.")
    cam.release()
    cv2.destroyAllWindows()
read_faces()
train_faces()
