import cv2
from ultralytics import YOLO
from collections import defaultdict
import os
# 设置环境变量 KMP_DUPLICATE_LIB_OK=TRUE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'path/to/platform/plugins'
model=YOLO('yolov8x.pt') 


cap = cv2.VideoCapture("C:/Users/user/Desktop/無人機影片/100MEDIA/3.30/DJI_0014.MOV")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)



fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("counting.mp4", fourcc, fps, size)


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

#登录复制

#视频帧循环
while cap.isOpened():
#读取一帧图像
    success, frame = cap.read()
    if success:
        #在帧上运行YOLOv8跟踪,persist为True表示保留跟踪信息,conf为0.3表示只检测置信值大于0.3的目标
        results = model.track(frame, conf=0.3, persist=True)
        #得到该帧的各个目标的ID
        if results[0].boxes.id !=None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            #遍历该帧的所有目标
            
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
                        #当前一帧在基准线的上面,当前帧在基准线的下面时,说明该车是从上往下运行
                        if h < size[1]-800 and y >= size[1]-800:                       
                            vehicle_out +=1                #out计数加1
                        if h > size[1]-800 and y <= size[1]-800:
                            vehicle_in +=1    
       #绘制基准线
        cv2.line(frame, (30,size[1]-800), (size[0]-30,size[1]-800), color=(25, 33, 189), thickness=2,lineType=4 )
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