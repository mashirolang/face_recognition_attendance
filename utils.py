from deepface import DeepFace
from deepface.basemodels import VGGFace
import pandas as pd
import os

model = VGGFace.loadModel()

def checkFace(frame):
    df = DeepFace.find(img_path = frame, db_path="./datasets", model_name = 'VGG-Face', model = model, distance_metric = 'cosine', enforce_detection = False)
    return df

def getIdentityId(identity):
    try:
        if len(os.path.split(identity)[-1].split(".")) > 1:
            return os.path.split(identity)[-1].split(".")[1]
        else:
            return None
    except:
        return None