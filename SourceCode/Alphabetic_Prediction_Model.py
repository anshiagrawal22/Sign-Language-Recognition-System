import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import string

# Loading trained model
model = load_model(r'Sign-Language-Recognition-System\Model\SignLang_CNN_Alphabet_Model.h5')
labels = [str(letter) for letter in string.ascii_uppercase]  

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    original_img = img.copy()  
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        handType = hand['type']  # 'Left' or 'Right'

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        height, width, _ = img.shape
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, width)
        y2 = min(y + h + offset, height)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size > 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Model Prediction
            img_input = cv2.resize(imgWhite, (64, 64))
            img_input = img_input.astype('float32') / 255.0
            img_input = np.expand_dims(img_input, axis=0)

            prediction = model.predict(img_input)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Drawing: clean and non-clashing
            label_text = f"{labels[predicted_class]}"
            conf_text = f"{confidence * 100:.2f}%"
            hand_text = f"{handType} Hand"

            # Draw on main image
            cv2.putText(img, f"{label_text}", (x1, y1 - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  #green --prediction number
            cv2.putText(img, f"{conf_text}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2) #cyan -- confidence 

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Show processed windows
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img)    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    chart_img = cv2.imread("Sign-Language-Recognition-System\American Sign Language Chart.jpg")
    cv2.imshow("American Sign Language Chart", chart_img)

cap.release()
cv2.destroyAllWindows()
