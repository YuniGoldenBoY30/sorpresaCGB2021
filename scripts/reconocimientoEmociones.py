import os

import cv2
import numpy as np


def recognize_emotions(data_path, emotion_name):
    print('Reconociendo emoción ....')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read('modeloLBPH.xml')

    video_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, image_frame = video_stream.read()
        if not ret: break
        image_frame = cv2.flip(image_frame, 1)

        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        aux_image_frame = gray.copy()

        capture_frame = cv2.hconcat([image_frame, np.zeros((480, 300, 3), dtype=np.uint8)])

        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_detected = aux_image_frame[y:y + h, x:x + w]
            face_detected = cv2.resize(face_detected, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = recognizer.predict(face_detected)
            cv2.putText(image_frame, f'{result}', (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 60:
                cv2.putText(image_frame, 'Sorpresa', (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image_emoji = cv2.imread(os.path.join(data_path, emotion_name, 'Emojis', 'sorpresa.jpeg'))
                capture_frame = cv2.hconcat([image_frame, image_emoji])
            else:
                cv2.putText(image_frame, 'No identificado', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                image_emoji = cv2.imread(os.path.join(data_path, emotion_name, 'Emojis', 'known.jpeg'))
                capture_frame = cv2.hconcat([image_frame, image_emoji])
        cv2.imshow('Camera', capture_frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    video_stream.release()
    cv2.destroyAllWindows()
