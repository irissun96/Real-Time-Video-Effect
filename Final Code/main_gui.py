import filter_functions as filters
import tkinter as tk
import dlib
import numpy as np
import cv2
from PIL import Image,ImageTk

# Define TK function
def fun_exit():
  global cap
  cap.release()

# Define TK root
top = tk.Tk()
top.title("Real Time Video effect")

# Define TK variable
CheckVar1 = tk.IntVar()
CheckVar1.set(0)
effect_var = tk.IntVar()
effect_var.set(0)

# Define widgets
panel = tk.Label(top)  # image panel
C1 = tk.Checkbutton(top, text = 'Enable Eye Blink',
                    variable = CheckVar1,
                    onvalue = 1, offvalue = 0,
                    font=("Arial ", 12, "bold"))
R0 = tk.Radiobutton(top, text="Original",
                    variable=effect_var, value=0,
                    font=("Arial ", 12, "bold"))
R1 = tk.Radiobutton(top, text="Sketch",
                    variable=effect_var, value=1,
                    font=("Arial ", 12, "bold"))
R2 = tk.Radiobutton(top, text="Black White",
                    variable=effect_var, value=2,
                    font=("Arial ", 12, "bold"))
R3 = tk.Radiobutton(top, text="Cartoon",
                    variable=effect_var, value=3,
                    font=("Arial ", 12, "bold"))
R4 = tk.Radiobutton(top, text="RGB Glitch",
                    variable=effect_var, value=4,
                    font=("Arial ", 12, "bold"))
R5 = tk.Radiobutton(top, text="Video Shadow",
                    variable=effect_var, value=5,
                    font=("Arial ", 12, "bold"))
R6 = tk.Radiobutton(top, text="Face Sticker",
                    variable=effect_var, value=6,
                    font=("Arial ", 12, "bold"))
B1 = tk.Button(top, text ="Exit", command = fun_exit,
                    font=("Arial ", 12, "bold"))

# Place buttons
panel.grid(row=0,column=0,rowspan=9)
C1.grid(row=0,column=1,sticky=tk.W)
R0.grid(row=1,column=1,sticky=tk.W)
R1.grid(row=2,column=1,sticky=tk.W)
R2.grid(row=3,column=1,sticky=tk.W)
R3.grid(row=4,column=1,sticky=tk.W)
R4.grid(row=5,column=1,sticky=tk.W)
R5.grid(row=6,column=1,sticky=tk.W)
R6.grid(row=7,column=1,sticky=tk.W)
B1.grid(row=8,column=1,sticky=tk.W+tk.E)


cap = cv2.VideoCapture(0)

# create objects
[ok, frame] = cap.read()
# frame = cv2.imread("../Image/Lenna_(test_image).png",1)
shadow = filters.videoShadow(buffer_size=16, delay_frames=3, alpha=0.5, frame_size=frame.shape)
faceSticker = filters.faceSticker_landmarks("../Image/sticker_file_0.txt")


NUM_FIlTER = 7
COUNTER = 0
blinked = False

# to detect the facial region
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


while cap.isOpened():
    
    [ok, frame] = cap.read()
    # frame = cv2.imread("../Image/Lenna_(test_image).png",1)
    frame = cv2.flip(frame, 1)
    frame_out = np.copy(frame)

    # apply effect 1 to 5
    i = effect_var.get()
    if i == 1:
        frame_out = filters.Sketch(frame_out)
    elif i == 2:
        frame_out = filters.BlackWhite(frame_out)
    elif i == 3:
        frame_out = filters.Cartoon(frame_out)
    elif i == 4:
        frame_out = filters.filterRGBGlitch_Random(frame_out)
    elif i == 5:
        frame_out = shadow.newFrame(frame_out)

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        # get the facial landmarks
        landmarks = np.array([[p.x, p.y] for p in predictor(frame, rect).parts()])
        blinked_rect, COUNTER = filters.detect_eye_blink(landmarks, COUNTER)
        # if blinked_rect:
        #     print("blinked")
        blinked = blinked or blinked_rect
        # apply effect 6
        if i == 6:
            frame_out = faceSticker.newFrame(frame_out, landmarks)

    # change video effect using eye blink
    if CheckVar1.get() == 1:
        if blinked:
            i += 1
            blinked = False
            # print(i)
        if i >= NUM_FIlTER:
            i = 0
    else:
        blinked = False
    effect_var.set(i)

    cv2image = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGBA)
    current_image = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=current_image)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    top.update()
