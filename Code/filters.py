import cv2
import numpy as np
import random



def filterGrayScale(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        frame[:,:,i] = frame_gray
    return frame

def filterRGBGlitch(frame, dis):
    frame[:,:,0:2] = np.concatenate((frame[:,-dis:,0:2], frame[:,:-dis,0:2]), axis=1)
    frame[:,:,2] = np.concatenate((frame[:,dis:,2], frame[:,:dis,2]), axis=1)
    return frame

def filterRGBGlitchRandom(frame):
    if random.randint(1,10) > 5:
        h, w, c = frame.shape
        dis = int(random.random() * 0.01 * w)
        frame = filterRGBGlitch(frame, dis)
    return frame