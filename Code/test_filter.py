import cv2
import time
import filters
import random

cap = cv2.VideoCapture(0)

print("Switch to video window. Then press 'p' to save image, 'q' to quit")

while cap.isOpened():

    [ok, frame] = cap.read()
    frame = cv2.flip(frame, 1)

    # filters
    # ===================

    frame = filters.filterGrayScale(frame)

    frame = filters.filterRGBGlitchRandom(frame)




    # ===================
    cv2.imshow('Live video', frame)

    key = cv2.waitKey(1)

    if key == ord('p'):
        cv2.imwrite('photo_{}.jpg'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()) ), frame)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

