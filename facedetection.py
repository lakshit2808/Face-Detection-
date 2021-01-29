import cv2 , time

video = cv2.VideoCapture(0) # it is used to capture videos here 0 means we are using laptops webcam no external webcam
x = 1
first_frame = None
while True:
    x = x + 1 # this will count the number of frames
    check , img = video.read() # this will give the frame as well as check if is running or not
    #print(check)
    #print(frame)

    time.sleep(.0000001) # this means every frame will change in 0.000001 second

    #img = cv2.imread("photo.jpg")
    img_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # loading a file which will have functions to detect face
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # this methon convert color img into gray

    face = img_cascade.detectMultiScale(gray_img , scaleFactor=1.1 , minNeighbors=8) # this method is used to detect a face here scalefactor means the accuracy by which it will detect the face and minneighbors means taking pixel which are near the point
    #print(face) OUTPUT: [[ 516  377 1024 1024]]
    if first_frame is None:
        first_frame = gray_img
        continue
    for x , y , z , a in face:
        det_img = cv2.rectangle(img , (x,y) ,(x+z , y+a) , (0,250,23) , 3) # here we are coverting the detected img into rectanlge and assining point for the img  here(0,255,0) is the color of rect and 3 is width of rect

    img_resize_det = cv2.resize(det_img , (650 , 650))

    cv2.imshow("Detected img" , img_resize_det)

    key =cv2.waitKey(1)
    if key == ord('q'):
        break



print(x)
video.release() # this is to stop the video

cv2.destroyAllWindows()