import cv2
import time
import filters
import random
from datetime import datetime

cap = cv2.VideoCapture(0)

print("Switch to video window. Then press 'p' to save image, 'q' to quit")


while cap.isOpened():
    
    [ok, frame] = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('Original', frame)

    # filters
    # ===================
    time_start = datetime.now()
    # frame = filters.filterGrayScale(frame)#Delay time: 0:00:00.001995
    
    # frame = filters.filterRGBGlitchRandom(frame)#Delay time: 0:00:00.004999

    frame = filters.skinSmooth_Bilateral_AlphaBlending(frame) # Delay time: 0:00:00.135920

    # frame = filters.skinSmooth_Gaussian_AlphaBlending(frame) # Delay time: 0:00:00.101941

    time_end = datetime.now()
    # ===================
    cv2.imshow('filtered', frame)
    
    key = cv2.waitKey(1)

    if key == ord('p'):
        cv2.imwrite('photo_{}.jpg'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()) ), frame)
    elif key == ord('q'):
        break

print("Delay time:", time_end - time_start)
cv2.destroyAllWindows()

