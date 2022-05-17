# adapated from tutorial https://www.youtube.com/watch?v=mPCZLOVTEc4
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
while True:
  ret, frame = cap.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Return the location of all of the faces in terms of positions
  # First parameter is scale factor (make value smaller for better accuracy)
  # Second paramter = how many neighbors each candidate rectangle should have to retun it (higher values = less detections)
  
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0, 0), 5)
    # Pass face to eye classifier 
    roi_gray = gray[y:y+w, x:x+w] #rows then columns
    roi_color = frame[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes: 
      cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0,255,0), 5)

  cv2.imshow('frame', frame)

  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()