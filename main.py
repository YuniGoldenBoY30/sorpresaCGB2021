import os

from scripts.capturandoRostros import capture_faces
from scripts.entrenando import training
from scripts.reconocimientoEmociones import recognize_emotions


def main():
    print('Starting app...')
    emotion_name = 'Sorpresa'
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emotions_path = os.path.join(data_path, emotion_name, f'Data{emotion_name}')
    if not os.path.exists(emotions_path):
        print('Carpeta creada: ', emotions_path)
        os.makedirs(emotions_path)

    faces_list = os.listdir(emotions_path)

    capture_faces(emotions_path)
    training(emotions_path, faces_list)
    recognize_emotions(data_path, emotion_name)

    print('Ending app...')


main()
