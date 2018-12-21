import cv2
import numpy as np
import random
import math
import dlib
from scipy.spatial import distance as dist

RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3

def eye_aspect_ratio(eye):
    # compute the euclidean distance between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # compute the EAR
    ear = (A + B) / (2 * C)
    return ear

def detect_eye_blink(landmarks, COUNTER):
	# get the left eye landmarks
    left_eye = landmarks[LEFT_EYE_POINTS]
    # get the right eye landmarks
    right_eye = landmarks[RIGHT_EYE_POINTS]
    # draw contours on the eyes
    # left_eye_hull = cv2.convexHull(left_eye)
    # right_eye_hull = cv2.convexHull(right_eye)
    # cv2.drawContours(frame, [left_eye_hull], -1, (224, 80, 119), 1) # (image, [contour], all_contours, color, thickness)
    # cv2.drawContours(frame, [right_eye_hull], -1, (224, 80, 119), 1)
    # compute the EAR for the left eye
    ear_left = eye_aspect_ratio(left_eye)
    # compute the EAR for the right eye
    ear_right = eye_aspect_ratio(right_eye)
    # compute the average EAR
    ear_avg = (ear_left + ear_right) / 2.0
    # detect the eye blink
    blinked = False
    if ear_avg < EYE_AR_THRESH:
        COUNTER += 1
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
        	blinked = True
        COUNTER = 0
    return blinked, COUNTER


def alphaBlending_2(frame_back, frame_front, frame_alpha, x=0, y=0):
    """
    Args:
        frame_back (np.array): the larger background BGR image with shape (h1, w1, 3)
        frame_front (np.array): the smaller frontground BGR image with shape (h2, w2, 3)
        frame_alpha (np.array): the alpha image with shape (h2, w2)
        x (int): the x coordinate of the starting pixel at frame_back
        y (int): the y coordinate of the starting pixel at frame_back
    Returns:
        np.array : the blended BGR image with shape (h1, w1, 3)
    """
    xf1 = 0
    yf1 = 0
    xf2 = frame_front.shape[1]
    yf2 = frame_front.shape[0]

    xb1 = x
    yb1 = y
    xb2 = x + xf2
    yb2 = y + yf2

    h, w = frame_back.shape[:2]
    if xb1 < 0:
        xf1 = -xb1
        xb1 = 0
    if yb1 < 0:
        yf1 = -yb1
        yb1 = 0
    if xb2 > w:
        xf2 -= xb2-w
        xb2 = w
    if yb2 > h:
        yf2 -= yb2-h
        yb2 = h

    alpha_front = frame_alpha[yf1:yf2, xf1:xf2] / 255.0
    alpha_back = 1.0 - alpha_front

    for c in range(3):

        frame_back[yb1:yb2, xb1:xb2, c] = \
            (alpha_front * frame_front[yf1:yf2, xf1:xf2, c] + alpha_back * frame_back[yb1:yb2, xb1:xb2, c])
    return frame_back


class videoShadow:
    """
    Video effect: add shadow to movement
    """
    def __init__(self, buffer_size, delay_frames, alpha, frame_size):
        self.buffer_size = buffer_size
        self.buffer = [ np.zeros(frame_size, dtype=np.uint8) for i in range(buffer_size) ]
        self.kr = 0  # read 
        self.kw = delay_frames  # write
        self.alpha = alpha

    def newFrame(self, frame):
        frame_out = cv2.addWeighted(frame, self.alpha, self.buffer[self.kr], 1-self.alpha,0)
        self.buffer[self.kw] = frame_out
        self.kr += 1
        if self.kr == self.buffer_size:
            self.kr = 0
        self.kw += 1
        if self.kw == self.buffer_size:
            self.kw = 0
        return frame_out


class faceSticker_landmarks():
    """
    Using dlib landmarks. Add face sticker with different mouth size
    """
    def __init__(self, sticker_file):
        self.changeSticker(sticker_file)

    def changeSticker(self,sticker_file):
        file = open(sticker_file, "r") 
        self.sticker_4ch = cv2.imread(file.readline().rstrip("\n"), cv2.IMREAD_UNCHANGED)
        # self.sticker_path = file.readline().rstrip("\n")
        self.sticker_mouth_0 = cv2.imread(file.readline().rstrip("\n"), cv2.IMREAD_UNCHANGED)
        self.sticker_mouth_1 = cv2.imread(file.readline().rstrip("\n"), cv2.IMREAD_UNCHANGED)
        self.sticker_mouth_2 = cv2.imread(file.readline().rstrip("\n"), cv2.IMREAD_UNCHANGED)
        self.sticker_scale = float(file.readline())
        self.mouth_scale = float(file.readline())
        self.mouth_offset = int(file.readline())

    def newFrame(self, frame, landmarks):
        sh, sw = self.sticker_4ch.shape[:2]
        sticker_4ch = np.copy(self.sticker_4ch)

        # add mouth
        mouth_l = landmarks[48]
        mouth_r = landmarks[54]
        mouth_u = landmarks[62]
        mouth_d = landmarks[66]

        mw = int(dist.euclidean(mouth_l, mouth_r)*self.mouth_scale)
        mh = int(max(dist.euclidean(mouth_u, mouth_d), 5)*self.mouth_scale)

        sticker_mouth_trans_0 = cv2.resize(self.sticker_mouth_0, (mw, int(mw/self.sticker_mouth_0.shape[1]*self.sticker_mouth_0.shape[0])), interpolation=cv2.INTER_CUBIC)
        sticker_mouth_trans_1 = cv2.resize(self.sticker_mouth_1, (mw, mh), interpolation=cv2.INTER_CUBIC)
        sticker_mouth_trans_2 = cv2.resize(self.sticker_mouth_2, (mw, int(mw/self.sticker_mouth_2.shape[1]*self.sticker_mouth_2.shape[0])), interpolation=cv2.INTER_CUBIC)

        sticker_mouth_4ch_trans = np.concatenate((sticker_mouth_trans_0, sticker_mouth_trans_1), axis=0)
        sticker_mouth_4ch_trans = np.concatenate((sticker_mouth_4ch_trans, sticker_mouth_trans_2), axis=0)
        
        x = int(sticker_4ch.shape[1]/2-sticker_mouth_4ch_trans.shape[1]/2)
        y = self.mouth_offset

        alphaBlending_2(sticker_4ch[:,:,:3], sticker_mouth_4ch_trans[:,:,:3], 
            sticker_mouth_4ch_trans[:,:,3], x, y)

        # rotate
        eye_left = landmarks[36]
        eye_right = landmarks[45]
        angle = -math.atan2(eye_left[1]-eye_right[1], eye_left[0]-eye_right[0])/math.pi*180
        if angle > 90:
            angle = angle - 180
        elif angle < -90:
            angle = angle + 180
        M = cv2.getRotationMatrix2D((sw/2,sw/2), angle, 1)
        sticker_4ch_trans = cv2.warpAffine(sticker_4ch, M, (sw,sh))

        # resize
        w = landmarks[16][0] - landmarks[0][0]
        # w = dist.euclidean(landmarks[16], landmarks[0])
        sticker_size = (int(w*self.sticker_scale), int(w/sw*sh*self.sticker_scale))
        sticker_4ch_trans = cv2.resize(sticker_4ch_trans, sticker_size, interpolation=cv2.INTER_CUBIC)
        sticker = sticker_4ch_trans[:,:,:3]
        sticker_alpha = sticker_4ch_trans[:,:,3]
        x = int(landmarks[29][0]-sticker_size[0]/2)
        y = int(landmarks[29][1]-sticker_size[1]/2)
        frame = alphaBlending_2(frame, sticker, sticker_alpha, x, y)
        return frame

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
    img_bw = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2BGR)
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

def filterRGBGlitch(frame, dis):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    frame[:,:,0:2] = np.concatenate((frame[:,-dis:,0:2], frame[:,:-dis,0:2]), axis=1)
    frame[:,:,2] = np.concatenate((frame[:,dis:,2], frame[:,:dis,2]), axis=1)
    return frame

def filterRGBGlitch_Random(frame):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    if random.randint(1,10) > 5:
        h, w, c = frame.shape
        dis = int(random.random() * 0.01 * w)
        frame = filterRGBGlitch(frame, dis)
    return frame