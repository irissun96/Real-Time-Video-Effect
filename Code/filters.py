import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import math


def filterGrayScale(frame):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        frame[:,:,i] = frame_gray
    return frame

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

def alphaBlending(frame_back, frame_front, frame_alpha):
    """
    Args:
        frame_back (np.array): the background BGR image with shape (h, w, c)
        frame_front (np.array): the frontground BGR image with shape (h, w, c)
        frame_alpha (np.array): the alpha image with shape (h, w)
    Returns:
        np.array : the blended BGR image with shape (h, w, c)
    """
    frame_alpha = frame_alpha.astype(float)/255
    frame_front = frame_front.astype(float)
    frame_front = cv2.multiply(frame_alpha, frame_front)
    frame_back = frame_back.astype(float)
    frame_back = cv2.multiply(1-frame_alpha, frame_back)
    frame_out = cv2.add(frame_front, frame_back)/255
    return frame_out

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
    x1 = x
    y1 = y
    x2 = x + frame_front.shape[1]
    y2 = y + frame_front.shape[0]

    alpha_front = frame_alpha / 255.0
    alpha_back = 1.0 - alpha_front

    for c in range(3):
        frame_back[y1:y2, x1:x2, c] = \
            (alpha_front * frame_front[:, :, c] + alpha_back * frame_back[y1:y2, x1:x2, c])
    return frame_back


def skinDetection_HSV(frame):
    """
    Better than skinDetection_RGB_H_CbCr
    ------
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the skin mask (False = 0, True = 255)
    """
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, lower, upper)
    return mask

def skinDetection_RGB_H_CbCr(frame):
    """
    From http://pesona.mmu.edu.my/~johnsee/research/papers/files/rgbhcbcr_m2usic06.pdf
    ------
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the skin mask (False = 0, True = 255)
    """
    def ruleA(R, G, B):
        # daylight illumination rule
        mask1 = R > 95
        mask1 = np.logical_and(mask1, G > 40)
        mask1 = np.logical_and(mask1, B > 20)
        mask1 = np.logical_and(mask1, (np.maximum(R,np.maximum(G,B)) - np.minimum(R, np.minimum(G,B))) > 15)
        mask1 = np.logical_and(mask1, abs(R-G) > 15)
        mask1 = np.logical_and(mask1, R > G)
        mask1 = np.logical_and(mask1, R > B)

        # flashlight or daylight lateral illumination rule
        mask2 = R > 220
        mask2 = np.logical_and(mask2, G > 210)
        mask2 = np.logical_and(mask2, B > 170)
        mask2 = np.logical_and(mask2, abs(R-G) <= 15)
        mask2 = np.logical_and(mask2, R > B)
        mask2 = np.logical_and(mask2, G > B)

        return np.logical_or(mask1, mask2)

    def ruleB(Cr, Cb):
        mask = Cr <= 1.5862 * Cb + 20 
        mask = np.logical_and(mask, Cr >= 0.3448 * Cb + 76.2069)
        mask = np.logical_and(mask, Cr >= -4.5652 * Cb + 234.5652)
        mask = np.logical_and(mask, Cr <= -1.15 * Cb + 301.75)
        mask = np.logical_and(mask, Cr <= -2.2857 * Cb + 432.85)
        mask = np.logical_and(mask, Cr >= 0.3448 * Cb + 76.2069)
        return mask

    def ruleC(H):
        ret, mask = cv2.threshold(H, 25, 230, cv2.THRESH_BINARY_INV)
        return mask

    B = np.array(frame[:,:,0])
    G = np.array(frame[:,:,1])
    R = np.array(frame[:,:,2])

    frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Y = frame_YCrCb[:,:,0]
    Cr = frame_YCrCb[:,:,1]
    Cb = frame_YCrCb[:,:,2]

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H = frame_hsv[:,:,0]
    # S = frame_hsv[:,:,1]
    # V = frame_hsv[:,:,2]

    maskA = ruleA(R, G, B)
    maskB = ruleB(Cr, Cb)
    maskC = ruleC(H)
    mask = maskA
    mask = np.logical_and(maskA, maskB)
    mask = np.logical_and(mask, maskC)
    mask = mask.astype(np.uint8) * 255
    return mask

def skinSmooth_Bilateral(frame):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    mask = skinDetection_HSV(frame)
    frame_mask = cv2.bitwise_and(frame, frame, mask = mask)
    frame_mask_inv = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
    frame_filtered = cv2.bilateralFilter(src=frame_mask, d=10, sigmaColor=50, sigmaSpace=50)
    frame_out = cv2.add(frame_mask_inv, frame_filtered)
    return frame_out
    

def skinSmooth_Bilateral_AlphaBlending(frame):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    mask = skinDetection_HSV(frame)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    alpha = cv2.GaussianBlur(mask,(5,5),0)
    frame_filtered = cv2.bilateralFilter(src=frame, d=10, sigmaColor=50, sigmaSpace=50)# Filter
    frame = alphaBlending(frame, frame_filtered, alpha)
    return frame

def skinSmooth_Gaussian_AlphaBlending(frame):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    mask = skinDetection_HSV(frame)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    alpha = cv2.GaussianBlur(mask,(5,5),0)
    frame_filtered = cv2.GaussianBlur(frame,(5,5),0)# Filter
    frame = alphaBlending(frame, frame_filtered, alpha)
    return frame

def faceEyeDetect(frame):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
    Returns:
        List[(x,y,w,h)]: face locations
        List[List[(x,y,w,h),...]]: eye locations on face
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    faces_eyes = []
    for (x,y,w,h) in faces:
        roi_gray = frame_gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes = sorted(eyes, key=lambda x:x[1])
        if len(eyes)>2:
            eyes = eyes[:2]
        faces_eyes.append(eyes)
    return faces, faces_eyes

def faceEyeDraw(frame, faces, faces_eyes):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
        faces (List[(x,y,w,h)]): face locations
        faces_eyes (List[List[(x,y,w,h),...]]): eye locations on face
    Returns:
        np.array : the filtered BGR image with shape (h, w, c)
    """
    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        for (ex,ey,ew,eh) in faces_eyes[i]:
                cv2.rectangle(frame,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(0,255,0),2)
    return frame

def faceSticker(frame, sticker_path, sticker_scale):
    """
    Args:
        frame (np.array): the orignal BGR image with shape (h, w, c)
        sticker_path (string): path to sticker file
    Returns:
        np.array : the BGR image with faceSticker shape (h, w, c)
    """
    sticker_4ch = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    sh, sw = sticker_4ch.shape[:2]
    faces, faces_eyes = faceEyeDetect(frame)
    for i in range(len(faces)):
        if len(faces_eyes[i]) == 2:
            center = []
            for (ex,ey,ew,eh) in faces_eyes[i]:
                center.append([(ex+ew)/2, (ey+eh)/2])
            angle = -math.atan2(center[0][1]-center[1][1], center[0][0]-center[1][0])/math.pi*180
            if angle > 90:
                angle = angle - 180
            elif angle < -90:
                angle = angle + 180
            M = cv2.getRotationMatrix2D((sw/2,sw/2), angle, 1)
            sticker_4ch_trans = cv2.warpAffine(sticker_4ch,M,(sw,sh))
        else:
            sticker_4ch_trans = sticker_4ch

        (x,y,w,h) = faces[i]
        sticker_size = (int(h/sh*sw*sticker_scale), int(h*sticker_scale))
        sticker_4ch_trans = cv2.resize(sticker_4ch_trans, sticker_size, interpolation=cv2.INTER_CUBIC)
        sticker = sticker_4ch_trans[:,:,:3]
        sticker_alpha = sticker_4ch_trans[:,:,3]
        try:
            x = int(x-(sticker_size[0]-w)/2)
            y = int(y-(sticker_size[1]-h)/2)
            frame = alphaBlending_2(frame, sticker, sticker_alpha, x, y)
        except:
            pass

        # for (ex,ey,ew,eh) in faces_eyes[i]:
        #         cv2.rectangle(frame,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(0,255,0),2)

    return frame

if __name__ == "__main__":
    frame = cv2.imread("../Image/test_0.jpg",1)
    cv2.imshow("Original", frame)

    time_start = datetime.now()

    # frame = skinSmooth_Bilateral_AlphaBlending(frame)
    # frame = cv2.bilateralFilter(src=frame, d=10, sigmaColor=50, sigmaSpace=50)

    # faces, faces_eyes = faceEyeDetect(frame)
    # frame = faceEyeDraw(frame, faces, faces_eyes)

    frame = faceSticker(frame, "../Image/emoji_resized_0.png")


    time_end = datetime.now()
    print("Delay time:", time_end - time_start)
    # plt.imshow(frame, cmap='gray')
    # plt.show()

    cv2.imshow("filtered", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


