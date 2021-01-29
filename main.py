import cv2

img = cv2.imread("photo.jpg")
img_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # loading a file which will have functions to detect face
gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # this methon convert color img into gray
"""img_resize = cv2.resize(gray_img , (650 , 1000))
cv2.imshow("Gray" , img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
face = img_cascade.detectMultiScale(gray_img , scaleFactor=1.1 , minNeighbors=8) # this method is used to detect a face here scalefactor means the accuracy by which it will detect the face and minneighbors means taking pixel which are near the point
#print(face) OUTPUT: [[ 516  377 1024 1024]]

for x , y , z , a in face:
    det_img = cv2.rectangle(img , (x,y) ,(x+z , y+a) , (0,250,23) , 3) # here we are coverting the detected img into rectanlge and assining point for the img  here(0,255,0) is the color of rect and 3 is width of rect

img_resize_det = cv2.resize(det_img , (650 , 1000))

cv2.imshow("Detected img" , img_resize_det)
cv2.waitKey(0)
cv2.destroyAllWindows()

