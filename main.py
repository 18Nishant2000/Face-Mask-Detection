import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
model = load_model('saved_model2')
size = (100, 100)

while True:
    _ , img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    img1 = []
    temp = cv2.resize(gray, size)
    img1.append(temp)
    img1 = np.array(img1)
    # img1 = img1/255
    img1 = img1.reshape(len(img1), 100, 100, 1)
    pred = model.predict(img1)
    result = ''
    print(pred)
    for i in pred:
        if i[0] > i[1]:
            result = 'With Mask'
        else:
            result = 'Without Mask'
    
    cv2.putText(img, result, size, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', img)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()