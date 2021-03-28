import cv2
import numpy as np
import dlib

#0 for !st webcam (index 1 for second web cam and so on)
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:

    #get frame from camera
    _, frame = cap.read() 

    #converting frame to grayframe frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    #for face detection
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    landmarks = predictor(gray, face)

    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    #name of the window frame
    cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    if key == 27:
        break