from retinaface import RetinaFace
import cv2
import os

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

student_id = input("Enter Your Student ID: ")
student_id = int(student_id)

count = 1

while True:
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video

    ret,frame = video.read()

    resp = RetinaFace.detect_faces(frame)

    if type(resp) is dict:
        print(resp['face_1']['facial_area'])

        x, y, w, h = resp['face_1']['facial_area']

        newpath = "datasets/" + str(student_id)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        cv2.imwrite("datasets/" + str(student_id) + "/User."+ str(student_id) + "." + str(count) + ".jpg", frame[(y-50):(h+50), (x-50):(w+50)])

        cv2.rectangle(frame, (x-50,y-50), (w+50, h+50), (118,181,197), 1)

        count = count + 1

    else:
        print("NO FACES")


    cv2.imshow("Frame", frame)

    if k == 27:
        break

    if count > 5:
        break

video.release()
cv2.destroyAllWindows()
print("Data added!")

