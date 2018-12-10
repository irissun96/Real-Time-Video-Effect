import cv2
import time
import filters
import random
from datetime import datetime

cap = cv2.VideoCapture(0)

[ok, frame] = cap.read()
shadow = filters.videoShadow(buffer_size=16, delay_frames=3, alpha=0.5, frame_size=frame.shape)
faceSticker = filters.faceSticker_dlib("../Image/sticker_file_0.txt")

print("Switch to video window. Then press 'p' to save image, 'q' to quit")


while cap.isOpened():
    
    [ok, frame] = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('Original', frame)

    # filters
    # ===================
    time_start = datetime.now()
    # frame = filters.filterGrayScale(frame)#Delay time: 0:00:00.001995
    
    # frame = filters.filterRGBGlitch_Random(frame)#Delay time: 0:00:00.004999

    # frame = filters.skinDetection_HSV(frame)

    # frame = filters.skinSmooth_Bilateral_AlphaBlending(frame) # Delay time: 0:00:00.135920

    # frame = filters.skinSmooth_Gaussian_AlphaBlending(frame) # Delay time: 0:00:00.101941

    # frame = filters.faceEyeDetect(frame) # Delay time: 0:00:00.212875
    # faces, faces_eyes = filters.faceEyeDetect(frame)
    # frame = filters.faceEyeDraw(frame, faces, faces_eyes)

    # frame = filters.faceSticker(frame, "../Image/Poop_Emoji_resized.png", 1.5) # Delay time: 0:00:00.167900

    frame = faceSticker.newFrame(frame)

    # frame = shadow.newFrame(frame)


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

