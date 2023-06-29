import cv2
import os

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

student_id = input("Enter Your Student ID: ")
student_id = int(student_id)

count = 0
data = 1

while True:
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video

    ret,frame = video.read()

    faces = face_detect.detectMultiScale(frame, 1.05, 10)

    for (x,y,w,h) in faces:
        count = count + 1

        newpath = "datasets/" + str(student_id)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if k == ord('q'):
            cv2.imwrite("datasets/" + str(student_id) + "/User."+ str(student_id) + "." + str(data) + ".jpg", frame[(y-50):(y+h+50), (x-50):(x+w+50)])
            data = data + 1

        cv2.rectangle(frame, (x-50,y-70), ((x+w)+50, (y+h)+50), (118,181,197), 1)


    cv2.imshow("Frame", frame)

    if k == 27:
        break

    if data > 5:
        break

    # if count > 100:
    #     break

video.release()
cv2.destroyAllWindows()
print("Data added!")

