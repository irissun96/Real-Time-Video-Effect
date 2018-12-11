import cv2
import numpy as np

path = "../Image/Poop_Emoji.png"
img = cv2.imread(path,-1)
h, w = img.shape[:2]
new_wh = int((h**2 + w**2)**0.5)
img_new = np.zeros([new_wh, new_wh, 4])
x = int((new_wh - w)/2)
y = int((new_wh - h)/2)
img_new[y:y+h,x:x+w,:] = img
cv2.imwrite(path[:-4]+"_resized"+path[-4:], img_new)
