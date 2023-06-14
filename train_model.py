import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path="datasets"

def getImageID(path):
    id_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    faces = []
    student_ids = []

    for id_path in id_paths:
        image_paths = [os.path.join(id_path, f) for f in os.listdir(id_path)]

        for image_path in image_paths:
            face_image = Image.open(image_path).convert('L')
            face_NP = np.array(face_image)
            student_id = (os.path.split(image_path)[-1].split(".")[1])
            student_id = int(student_id)

            faces.append(face_NP)
            student_ids.append(student_id)
            cv2.imshow("Training", face_NP)
            cv2.waitKey(1)
        
    return student_ids, faces

student_ids, faces = getImageID(path)
recognizer.train(faces, np.array(student_ids))
recognizer.write("Model.yml")
cv2.destroyAllWindows()
print("Updated Model Created!!")