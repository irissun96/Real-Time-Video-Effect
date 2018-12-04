import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime


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
    alpha = alpha.astype(float)/255

    frame_filtered = cv2.bilateralFilter(src=frame, d=10, sigmaColor=50, sigmaSpace=50)# Filter
    frame_filtered = frame_filtered.astype(float)
    frame_filtered = cv2.multiply(alpha, frame_filtered)

    frame_background = frame.astype(float)
    frame_background = cv2.multiply(1-alpha, frame_background)
    frame_out = cv2.add(frame_filtered, frame_background)/255
    return frame_out

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
    alpha = alpha.astype(float)/255

    frame_filtered = cv2.GaussianBlur(frame,(5,5),0)# Filter
    frame_filtered = frame_filtered.astype(float)
    frame_filtered = cv2.multiply(alpha, frame_filtered)
    
    frame_background = frame.astype(float)
    frame_background = cv2.multiply(1-alpha, frame_background)
    frame_out = cv2.add(frame_filtered, frame_background)/255
    return frame_out



if __name__ == "__main__":
    frame = cv2.imread('test_image_1.jpg',1)
    cv2.imshow("Original", frame)

    time_start = datetime.now()
    frame = skinSmooth_Bilateral_AlphaBlending(frame)
    # frame = cv2.bilateralFilter(src=frame, d=10, sigmaColor=50, sigmaSpace=50)
    time_end = datetime.now()
    print("Delay time:", time_end - time_start)
    # plt.imshow(frame, cmap='gray')
    # plt.show()

    cv2.imshow("filtered", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


