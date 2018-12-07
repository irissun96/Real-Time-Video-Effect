import cv2
import os

# capture video from live video stream
cap = cv2.VideoCapture(0)

while True:
    # get the frame
    [ok, frame] = cap.read()
    frame = cv2.flip(frame,1)

    # num_down = 2         #缩减像素采样的数目
    # num_bilateral = 7    #定义双边滤波的数目
    #
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     #读取图片
    # #用高斯金字塔降低取样
    # frame_color = frame_rgb
    # for _ in range(num_down):
    #     frame_color = cv2.pyrDown(frame_color)
    # #重复使用小的双边滤波代替一个大的滤波
    # for _ in range(num_bilateral):
    #     frame_color = cv2.bilateralFilter(frame_color,d=9,sigmaColor=9,sigmaSpace=7)
    # #升采样图片到原始大小
    # for _ in range(num_down):
    #     frame_color = cv2.pyrUp(frame_color)
    # #转换为灰度并且使其产生中等的模糊
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # blur = cv2.medianBlur(gray, 7)
    # #检测到边缘并且增强其效果
    # img_edge = cv2.adaptiveThreshold(blur,255,
    #                                  cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                  cv2.THRESH_BINARY,
    #                                  blockSize=9,
    #                                  C=2)
    # #转换回彩色图像
    # edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # cartoon = cv2.bitwise_and(frame_color, edge)
    # #
    #



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur,255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  9,2)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    cv2.imshow('Cartoon Filter', cartoon)
    cv2.imshow('Original', frame)

    key = cv2.waitKey(1) & 0xFF
    # When key 'Q' is pressed, exit
    if key is ord('q'):
        break

cap.release()
cv2.destroyAllWindows()