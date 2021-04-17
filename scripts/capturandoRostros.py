import os

import cv2


def capture_faces(emotions_path):
    print("Capturando rostros...")
    video_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0

    while True:
        ret, image_frame = video_stream.read()
        if not ret: break

        image_frame = cv2.flip(image_frame, 1)
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        aux_image_frame = image_frame.copy()
        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_detected = aux_image_frame[y:y + h, x:x + w]
            face_detected = cv2.resize(face_detected, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(emotions_path + f'/rostro_{count + 1}.jpg', face_detected)
            count = count + 1
        cv2.rectangle(image_frame, (10, 5), (255, 25), (255, 0, 255), 1)
        cv2.putText(image_frame, f"Faltan {300 - count} rostros por tomar", (10, 20), 2, 0.5, (128, 0, 20))
        cv2.imshow('Imagen', image_frame)

    video_stream.release()
    cv2.destroyAllWindows()
    return print('Pasando a entrenar metodo LBPH')
