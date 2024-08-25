import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
from cap_from_youtube import cap_from_youtube
from cap_from_youtube import list_video_streams
from tkinter import *
from tkinter import messagebox






def mouse_callback_xy(event, x,y):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
    return x,y


def messagebox_xy():
    top = Tk()
    top.title("請輸入座標以描繪基準線")   
    label_x =Label(top, text = '請輸入x座標以繪製基準線')
    mouseback_x=Entry(top, bd=5)
    label_x.pack()
    mouseback_x.pack()
    mainloop()

messagebox_xy()

