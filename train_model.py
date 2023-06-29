import os
import time
from utils import checkFace

file_path = "./datasets/representations_vgg_face.pkl"

while os.path.exists(file_path):
    os.remove(file_path)
    time.sleep(1)
    print("Deleting the current model..")

checkFace("./test/User1.jpg")

print("Successfully created new model...")