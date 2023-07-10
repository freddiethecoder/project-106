import cv2


# Create our body classifier
full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    body = full_body_cascade.detectMultiScale(gray)
    print(body)
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in body:
         cv2.circle(frame,(x+(w/2),y+(h/2)),50,(0,95,255),5)

    cv2.imshow("Web cam",frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
