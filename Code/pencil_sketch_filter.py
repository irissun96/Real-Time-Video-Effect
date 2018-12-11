
import cv2
import numpy as np
from matplotlib import  pyplot as plt



def dodgeV2(image, mask):
  return cv2.divide(image, 255-mask, scale=256)

def Sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_inv = 255- gray
    blur = cv2.GaussianBlur(gray_inv, ksize=(21, 21),
                            sigmaX=0, sigmaY=0)
    img_sketch = dodgeV2(gray, blur)
    img_sketch = cv2.cvtColor(img_sketch, cv2.COLOR_GRAY2BGR)
    return img_sketch

def BlackWhite(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img_bw = cv2.cvtColor(img_bw, cv2.COLOR_GRAY2BGR)
    return img_bw

def Cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 9)
    edges = cv2.adaptiveThreshold(blur,255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  9,2)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

if __name__ == "__main__":
    # capture video from live video stream
    cap = cv2.VideoCapture(0)
    while True:
        # get the frame
        [ok, frame] = cap.read()
        frame = cv2.flip(frame,1)

        img_sketch = Sketch(frame)
        img_bw = BlackWhite(frame)
        img_cartoon = Cartoon(frame)

        cv2.imshow('Sketch Filter', img_sketch)
        cv2.imshow('B&W Filter', img_bw)
        cv2.imshow('Cartoon Filter', img_cartoon)
        cv2.imshow('Original', frame)

        key = cv2.waitKey(1) & 0xFF
        # When key 'Q' is pressed, exit
        if key is ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()