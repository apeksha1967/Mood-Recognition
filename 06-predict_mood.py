import cv2

dataset = cv2.CascadeClassifier('data.xml')

capture = cv2.VideoCapture(0)
facedata = []
while True:
    ret,img = capture.read()
    # print(ret)
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),4)

            face = gray[y:y+h, x:x+w]
    #        face = cv2.resize(face, (64,64))
            if len(facedata) < 40:
                facedata.append(face)
                print(len(facedata))

            cv2.imshow('result', img)
            cv2.imwrite('image.png', face)
        if cv2.waitKey(1) == 27 or len(facedata) >= 40:
            break
    else:
        print("Camera not working")

cv2.destroyAllWindows()
capture.release()

from keras.models import load_model

# load model
model = load_model('model.h5')

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('image.png', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

predicted_class_indices=np.argmax(result,axis=1)

labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

labels = dict((v,k) for k,v in labels.items())
prediction = [labels[k] for k in predicted_class_indices]
#print(prediction[0])

from tkinter import *
from PIL import Image, ImageTk

canvas_width = 270
canvas_height = 270

master = Tk()
master.minsize(200,150)

canvas = Canvas(master, width=canvas_width, height=canvas_height, bg = 'white')
canvas.pack()

image = "emojis/" + prediction[0] + ".png"

img = ImageTk.PhotoImage(Image.open(image))
canvas.create_image(40, 40, anchor=NW, image=img)

master.mainloop()
