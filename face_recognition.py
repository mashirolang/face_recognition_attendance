import cv2
from utils import checkFace, getIdentityId

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video

    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if (len(faces) > 0):
        x, y, w,h = faces[0]
        frame_height, frame_width = frame.shape[:2]

        if y-50 >= 0 and y+h+50 <= frame_height and x-50 >= 0 and x+w+50 <= frame_width:
            df = checkFace(frame[(y-50):(y+h+50), (x-50):(x+w+50)])
            cv2.rectangle(frame, (x-50,y-70), ((x+w)+50, (y+h)+50), (118,181,197), 1)
          
            if not df.empty:
                identity = df.iloc[0]["identity"]
                confidence = df.iloc[0]["VGG-Face_cosine"]

                print(getIdentityId(identity))

                if confidence < 0.15:
                    # Face Matched, proceed next step which is attendance tracker and placing it into database
                    print("Face Matched")
                else:
                    # Confidence too low, not accurate
                    print("Confidence too low")
                    

            else:
                # Not registered on datasets
                print("DF is empty")

        else:
            # The face is out of bounds
            print("Face out of bounds")
    
    cv2.imshow('Face Detection', frame)

    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
