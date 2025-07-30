import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Detect only one hand

offset = 20  # Margin around the bounding box
imgSize=300 

#saving images
folder= r"Sign-Language-Recognition-System\data\Numerical\9" #r- makes it raw string and python won't treat \0 as escape character
counter=0
number=0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Detect hand and draw landmarks
    img = cv2.flip(img, 1)  # Flip the image horizontally

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Bounding box coordinates
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255 #300x300


        height, width, _ = img.shape  # Get image size
        # Adjust x because image is flipped
        x = width - x - w
        # Ensure crop stays within image boundaries
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0) 
        x2 = min(x + w + offset, width)
        y2 = min(y + h + offset, height)
        # Crop the hand region
        imgCrop = img[y1:y2, x1:x2]
        
        imgCropShape=imgCrop.shape
        
        #centering images into the white box
        aspectRatio=h/w
        if aspectRatio>1: #height is greater
            k= imgSize/h #constant
            wcal=math.ceil(k*w) #always round off the decimal value to higher number
            imgResize=cv2.resize(imgCrop,(wcal,imgSize)) 
            imgResizeShape=imgResize.shape 
            
            wGap=math.ceil((imgSize-wcal)/2) #for keeping image in center
            
            imgWhite[0:imgResizeShape[0],wGap:wcal+wGap]=imgResize  #putting matrix img crop on img white
            
        else: #width is greater
            k= imgSize/w #constant
            hcal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hcal)) 
            imgResizeShape=imgResize.shape 
            
            hGap=math.ceil((imgSize-hcal)/2) #for keeping image in center
            
            imgWhite[hGap:hcal+hGap,:]=imgResize  #putting matrix img crop on img white
        
        if imgCrop.size > 0:
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img) 
    key = cv2.waitKey(1)
    # Exit when 'q' is pressed
    if  key == ord('q'):
        break
    elif key == ord('s'):
        counter += 1
        filepath = f"{folder}/Image_{number}.jpg"
        cv2.imwrite(filepath, imgWhite) #saves the images  
        print(f"Saved: {filepath} ({counter})")
        number += 1
