from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2
import os
 # 设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("C:/Users/user/Desktop/無人機影片/100MEDIA/學長姐/DJI_0001.MOV")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

line_pts = [(0, 360), (1280, 360)]

# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()