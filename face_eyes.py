import cv2
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
img = cv2.imread('lena.jpg')
cap=cv2.VideoCapture(0)
while (cap.isOpened()):
    ret,img=cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.1, 4)
    eyes = eye.detectMultiScale(gray,1.1,4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes = eye.detectMultiScale(roi_gray)
        for (x1, y1, w1, h1) in eyes:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 5)
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord("q"):
        break






cap.release()

cv2.destroyAllWindows()
