import os
import time

import cv2
import numpy as np


def training(emotions_path, faces_list):
    labels = []
    facesData = []
    label = 0
    for fileName in faces_list:
        labels.append(label)
        facesData.append(cv2.imread(os.path.join(emotions_path, fileName), 0))
        image = cv2.imread(os.path.join(emotions_path, fileName), 0)
        cv2.imshow('image', image)
        cv2.waitKey(50)
    label = label + 1
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando ...")
    start = time.time()
    recognizer.train(facesData, np.array(labels))
    train_time = time.time() - start
    print("Tiempo de entrenamiento : ", train_time)
    recognizer.write("modeloLBPH.xml")
    print('Pasando al reconocimiento de emociones...')
