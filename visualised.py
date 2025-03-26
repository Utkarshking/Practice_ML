import cv2

# create a camera object 
cam=cv2.VideoCapture(0)

# model initialisation
model=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:
    success,img=cam.read()
    if not success:
        print("Reading from Camera Failed")
    faces=model.detectMultiScale(img,1.3,5)

    for face in faces:
        x,y,w,h=face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    

    cv2.imshow("ImageWindow",img)

    # Pause for 1 ms before you read the next image
    key=cv2.waitKey(1)
    # if we press q the while loop will automatically break and it will stop taking the images
    if key==ord('q'):
        break


# Release Camera, and Destroy Window
cam.release()
cv2.destroyAllWindows()


