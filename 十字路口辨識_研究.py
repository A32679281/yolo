from tkinter import messagebox
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

prev_time = 0
targets = {}
count1 = 0    # 左下
count2 = 0    # 右下
count3 = 0    # 右上
count4 = 0    # 左上
dist_threshold = 150


#model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("DJI_0030.MOV") #input("請輸入網址：") #YT進不來  #"https://cctv.bote.gov.taipei:8501/mjpeg/068"
cv2.namedWindow("video")
cv2.setMouseCallback("video", mouse_callback)
while cap.isOpened():
    success, frame = cap.read()
    #cv2.line(frame, (175, 1250), (1650, 1900), (0, 0, 255), 5)   # 左下座標線

    #cv2.line(frame, (200, 1200), (748, 1500), (0, 0, 255), 5)
    #cv2.line(frame, (1050, 1590), (1590, 1845), (0, 0, 255), 5)
    if not success:
        messagebox.showinfo("錯誤提示","Ignoring empty camera frame.")
        break
    frame = cv2.resize(frame, (1240, 810))
    results = model(frame)
    #detections = results.xyxy[0]  # 獲取偵測到的物件和其邊界框座標

    print(results)
    prev_time = time.time()

    cv2.imshow("video", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''

    for detection in detections:
        class_idx = int(detection[-1].item())  # 獲取物件的類別標籤
        if class_idx in [0, 2, 3, 5, 7]:  # 只取汽車和卡車
            xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 繪製邊界框
            # 計算中心點座標並繪製
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            # print(center_x,center_y)
            center = (center_x,center_y)

    cv2.putText(frame, f'FPS: {int(1 / (time.time() - prev_time))}',
                (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
'''

 
#names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    
