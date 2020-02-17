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

            cv2.imwrite('image.png', face)
        if cv2.waitKey(1) == 27 or len(facedata) >= 40:
            break
    else:
        print("Camera not working")

cv2.destroyAllWindows()
capture.release()
