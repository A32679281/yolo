import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
from cap_from_youtube import cap_from_youtube
from cap_from_youtube import list_video_streams
from tkinter import * 
from tkinter import messagebox
import time
# 设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




click_coordinates={"x":None, "y":None}
coordinate_x,coordinaes_y=0,0
def mouse_callback(event, X, Y, flags, param):
    global coordinate_x,coordinaes_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coordinates["x"]=X
        click_coordinates["y"]=Y
        coordinate_x=X
        coordinaes_y=Y
        updatexy()



def updatexy():
    entry_x.delete(0,END)
    entry_y.delete(0,END)
    entry_x.insert(0,str(click_coordinates["x"]))
    entry_y.insert(0,str(click_coordinates["y"]))
model=YOLO('yolov8x.pt') 

youtube_url = 'https://www.youtube.com/watch?v=yPdfq9ZxQcQ'
cap = cap_from_youtube(youtube_url, '720p')
#cap=cv2.VideoCapture("C:/Users/user/Desktop/無人機影片/100MEDIA/學長姐/DJI_0001.MOV")
    #登录复制
cv2.namedWindow("YOLOv8 Tracking")

fps = cap.get(cv2.CAP_PROP_FPS)
#size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size = (1440, 810)
fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("counting2.mp4", fourcc, fps, size)


def close():
    top.destroy()
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows ()

def go_on():
    top.destroy()
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
#得到目标矩形框的左上角和右下角坐标
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    #绘制矩形框
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
    #得到要书写的文本的宽和长,用于给文本绘制背景色
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        #确保显示的文本不会超出图片范围
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2. LINE_AA)
        #书写文本
        cv2.putText(image,label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),0,2 / 3,txt_color,thickness=1,lineType=cv2. LINE_AA)



# track_history用于保存目标ID,以及它在各帧的目标位置坐标,这些坐标是按先后顺序存储的
track_history = defaultdict(lambda:[]) 
#车辆的计数变量
vehicle_in = 0
vehicle_out = 0

def messagebox_xy():
    global entry_x,entry_y,btn_confirm,btn_cancel,top
    top =Tk()
    top.title("點擊圖片以輸入座標")   
    window_width =top.winfo_screenwidth()
    window_height =top.winfo_screenheight()
    width = 220
    height = 75
    left_x = int((window_width - width)/2)       # 計算左上 x 座標
    top_y = int((window_height - height)/2) 
    top.geometry(f'{width}x{height}+{left_x}+{top_y}')
    label_x =Label(top, text = 'x座標: ',width=10)
    label_x.grid(row=0,column=0)
    entry_x=Entry(top,width=10)
    entry_x.grid(row=0,column=1)
    label_y =Label(top, text ='y座標:',width=10)
    label_y.grid(row=1,column=0)
    entry_y=Entry(top,width=10)
    entry_y.grid(row=1,column=1)
    btn_confirm=Button(top,text="確認",command=go_on,width=10)
    btn_confirm.grid(row=2,column=0,sticky=W)
    btn_cancel=Button(top,text="取消",command=close,width=10)
    btn_cancel.grid(row=2,column=1,sticky=W)
    top.mainloop()

if cap.isOpened():
    success, frame = cap.read()
    frame = cv2.resize(frame, (1440, 810)) 
    cv2.imshow("YOLOv8 Tracking",frame)
    cv2.setMouseCallback("YOLOv8 Tracking", mouse_callback)
    messagebox_xy()
    cv2.setMouseCallback("YOLOv8 Tracking", mouse_callback)
    messagebox_xy()
#视频帧循环

while cap.isOpened():
#读取一帧图像
    success, frame = cap.read()
   
    if success:
        frame = cv2.resize(frame, (1440, 810)) 
        #在帧上运行YOLOv8跟踪,persist为True表示保留跟踪信息,conf为0.3表示只检测置信值大于0.3的目标
        results = model.track(frame, conf=0.3, persist=True)
        #得到该帧的各个目标的ID
        if results[0].boxes.id !=None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            #遍历该帧的所有目标
            #登录复制
            for i in range(2):                   
                for track_id, box in zip(track_ids, results[0].boxes.data):
                    if box[-1] == 2: #目标为小汽车
                        #绘制该目标的矩形框
                        box_label(frame, box, '#'+str(track_id)+' car', (167, 146, 11))
                        #得到该目标矩形框的中心点坐标(x,y)
                        x1, y1, x2, y2 = box[:4]
                        x= (x1+x2)/2
                        y= (y1+y2)/2
                        #提取出该ID的以前所有帧的目标坐标,当该ID是第一次出现时,则创建该ID的字典
                        track = track_history[track_id]
                        track.append((float(x), float(y)))#追加当前目标ID的坐标
                        #只有当track中包括两帧以上的情况时,才能够比较前后坐标的先后位置关系
                        if len(track) > 1:
                            _, h= track[-2] #提取前一帧的目标纵坐标
                            #我们设基准线为纵坐标是size[1]-400的水平线
                            #当前一帧在基准当线的上面,前帧在基准线的下面时,说明该车是从上往下运行
                            if h < coordinaes_y and y >= coordinaes_y:                       
                                vehicle_out +=1                #out计数加1
                            if h > coordinaes_y and y <= coordinaes_y:
                                vehicle_in +=1      
    #绘制基准线
        cv2.line(frame, (0,coordinaes_y), (size[0]-30,coordinaes_y), color=(25, 33, 189), thickness=2,lineType=4 )
        cv2.line(frame, (coordinate_x,0), (coordinate_x,size[1]), color=(25, 33, 189), thickness=2,lineType=4 )
        #实时显示进、出车辆的数量
        cv2.putText(frame, 'north: '+str(vehicle_in), (595, size[1]-410),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'south: '+str(vehicle_out), (573, size[1]-370),cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #cv2.putText(frame, "https://blog.csdn.net/zhaocj", (25, 50),
        #cv2. FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        #frame = cv2.resize(frame, (1440, 810))     # 調整resize時在imshow的時候方便看，但無法輸出影片
        cv2.imshow("YOLOv8 Tracking",frame)#显示标记好的当前帧图像
        videoWriter.write(frame) #3A#
        if cv2.waitKey(1)& 0xFF == 27: #'q'按下时,终止运行
            break
    else: #视频播放结束时退出循环
        break

#释放视频捕捉对象,并关闭显示窗口
cap.release()
videoWriter.release()
cv2.destroyAllWindows ()