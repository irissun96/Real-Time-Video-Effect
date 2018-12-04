import cv2

def filterGrayScale(frame):
	return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
