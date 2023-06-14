import cv2
import os

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter Your Name: ")
student_id = input("Enter Your Student ID: ")
student_id = int(student_id)

count = 0

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count = count + 1

        newpath = "datasets/" + str(student_id)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        cv2.imwrite("datasets/" + str(student_id) + "/User."+ str(student_id) + "." + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (118,181,197), 1)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if count > 200:
        break

video.release()
cv2.destroyAllWindows()
print("Data added!")

