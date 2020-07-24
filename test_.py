import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

#--------------------------------------------------------------------
#                   To check on webcam
#--------------------------------------------------------------------
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("miscelleanous/haarcascade_frontalface_default.xml")
model = load_model(os.path.join(os.getcwd(), 'smile_detect'))
while True:
    ret, img = cap.read()
    if ret != None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) != 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = np.reshape(roi, (1, 64, 64, 1))/255.0
            pred = model.predict(roi)
            if pred[0][0] < pred[0][1]:
                cv2.putText(img, "Smiling", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            else:
                cv2.putText(img, "Not smiling", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            cv2.imshow("Window", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()

#---------------------------------------------------------------------------
#                               To check on test images
#---------------------------------------------------------------------------

# for root, _, files in os.walk(os.path.join(os.getcwd(), 'miscelleanous')):
#     for filename in files:
#         if ".jpg" in filename or ".jpeg" in filename:    
#             img = cv2.imread(os.path.join(root, filename))
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = cascade.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
#                 roi = gray[y:y + h, x:x + w]
#                 roi = cv2.resize(roi, (64, 64))
#                 roi = np.reshape(roi, (1, 64, 64, 1))/255.0
#                 pred = model.predict(roi)
#                 print(pred)
#                 cv2.imshow("Window", img)
#                 cv2.waitKey(0)
