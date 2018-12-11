import numpy as np
import cv2
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