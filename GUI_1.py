import customtkinter
from CTkMessagebox import CTkMessagebox
from tkinter.filedialog import askopenfile
import cv2
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from untitled0 import get_points, remove_duplicates, is_in_list
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tkinter.simpledialog as simpledialog

# Define variable
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def run_model(model,
        source="0",
        conf=0.55,  # 設定信心值
        iou=0,  # NMS IOU threshold
        classes=2,  # 只辨識汽車
        ):

    results = model.track(source=source, iou=iou, conf=conf, classes=classes, persist=True)
    for r in results:
        boxes = r.boxes
    return boxes

# 儲存熱圖
def save_heatmap_video_img(heatmap_list, save_featmap_video, save_heatmap_img, fps):
        print(">>>>>>>正在儲存熱圖影片及圖片")
        if not os.path.exists(save_featmap_video):
            os.makedirs(save_featmap_video)

        if not os.path.exists(save_heatmap_img):
            os.makedirs(save_heatmap_img)

        # 設定輸出圖片大小
        height, width, _ = heatmap_list[0].shape  # 取圖片大小

        out = cv2.VideoWriter(f"{save_featmap_video}/heatmap_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # 重複儲存每一幀圖片
        for frame in range(len(heatmap_list)):
            out.write(heatmap_list[frame])
            cv2.imwrite(f"{save_heatmap_img}/{frame}_heatmap.jpg", heatmap_list[frame])

        out.release()
        print(f">>>>>>>儲存完畢")
        print(f">>>>>>>圖片路徑: {save_heatmap_img}")
        print(f">>>>>>>影片路徑: {save_featmap_video}")


class main_window(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.FileName = ""
        self.IsOpenFile = False


       # self.model = YOLO("./weights/modelm.pt")   被下行yolov8.pt覆蓋
        self.model = YOLO("yolov8x.pt")
        self.track = []
        self.heatmap_count = 0

        self.e = 0 # 東
        self.s = 0 # 南
        self.w = 0 # 西
        self.n = 0 # 北
        self.dist_threshold = 150

        FontStyle = customtkinter.CTkFont(size=20, family="Arial")

        self.title("影像辨識與車流偵測")
        self.geometry(f"{1740}x{1020}")

        self.grid_columnconfigure((0), weight=3)
        self.grid_columnconfigure((1), weight=1)
        self.grid_rowconfigure((1), weight=2)

        self.ResultFrame = customtkinter.CTkFrame(self)
        self.ResultFrame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.label_Direction = customtkinter.CTkLabel(self.ResultFrame, text="方向", font=FontStyle)
        self.label_Direction.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_E = customtkinter.CTkLabel(self.ResultFrame, text="East", font=FontStyle)
        self.label_Direction_E.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_S = customtkinter.CTkLabel(self.ResultFrame, text="South", font=FontStyle)
        self.label_Direction_S.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_W = customtkinter.CTkLabel(self.ResultFrame, text="West", font=FontStyle)
        self.label_Direction_W.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Direction_N = customtkinter.CTkLabel(self.ResultFrame, text="North", font=FontStyle)
        self.label_Direction_N.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        self.label_Quantity = customtkinter.CTkLabel(self.ResultFrame, text="數量", font=FontStyle)
        self.label_Quantity.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_E = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_E.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_S = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_S.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_W = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_W.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Quantity_N = customtkinter.CTkLabel(self.ResultFrame, text="0", font=FontStyle)
        self.label_Quantity_N.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

        self.label_Congestion = customtkinter.CTkLabel(self.ResultFrame, text="擁擠度", font=FontStyle)
        self.label_Congestion.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_E = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_E.grid(row=1, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_S = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_S.grid(row=2, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_W = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_W.grid(row=3, column=3, padx=10, pady=10, sticky="nsew")

        self.label_Congestion_N = customtkinter.CTkLabel(self.ResultFrame, text="低", font=FontStyle)
        self.label_Congestion_N.grid(row=4, column=3, padx=10, pady=10, sticky="nsew")

        # Play Video Frame
        self.PlayVideoFrame = customtkinter.CTkFrame(self)
        self.PlayVideoFrame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nswe")

        self.label_Video = customtkinter.CTkLabel(self.PlayVideoFrame, text="請選擇影片", font=customtkinter.CTkFont(size=20, family="Arial"))
        self.label_Video.place(relx=0.5, rely=0.5, anchor="center")


        self.ResultGraphFrame = customtkinter.CTkFrame(self)
        self.ResultGraphFrame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nswe")

        self.label_GraphTitle = customtkinter.CTkLabel(self.ResultGraphFrame, text="各方向車流量", font=customtkinter.CTkFont(size=20, family="Arial", weight="bold"))
        self.label_GraphTitle.place(relx=0.5, rely=0.06, anchor='center')
        self.canvas = customtkinter.CTkCanvas(self.ResultGraphFrame, width=350, height=340)
        self.canvas.place(relx=0.5, rely=0.55, anchor="center")

        self.ButtonFrame = customtkinter.CTkFrame(self)
        self.ButtonFrame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        self.OpenFileBtn = customtkinter.CTkButton(self.ButtonFrame, text='選擇影片',
                                                   font=customtkinter.CTkFont(size=20, family="Arial"),
                                                   command=lambda:self.open_file())
        self.OpenFileBtn.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        self.StartBtn = customtkinter.CTkButton(self.ButtonFrame, text='雙向道',
                                               font=customtkinter.CTkFont(size=20, family="Arial"),
                                               command=lambda:self.start())
        self.StartBtn.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        self.Start1Btn = customtkinter.CTkButton(self.ButtonFrame, text='十字路口',
                                                font=customtkinter.CTkFont(size=20, family="Arial"),
                                                command=lambda: self.start1())
        self.Start1Btn.grid(row=0, column=2, padx=10, pady=10, sticky="n")

        self.OpenPathBtn = customtkinter.CTkButton(self.ButtonFrame, text='查看結果',
                                                   font=customtkinter.CTkFont(size=20, family="Arial"),
                                                   command=lambda: self.OpenFilePath())
        self.OpenPathBtn.grid(row=0, column=3, padx=10, pady=10, sticky="n")


    def get_user_input(self):
        self.location_e = simpledialog.askstring("輸入東向判別座標", "請輸入數值：\t\t\t\t\t")
        self.location_w = simpledialog.askstring("輸入西向判別座標", "請輸入數值：\t\t\t\t\t")
        self.location_s = simpledialog.askstring("輸入南向判別座標", "請輸入數值：\t\t\t\t\t")
        self.location_n = simpledialog.askstring("輸入北向判別座標", "請輸入數值：\t\t\t\t\t")

        if self.location_e is not None:
            values_e = self.location_e.split()  # 用空白字符拆分字串
            if len(values_e) == 4:
                try:
                    self.location_e1 = int(values_e[0])  # 第一個數值
                    self.location_e2 = int(values_e[1])  # 第二個數值
                    self.location_e3 = int(values_e[2])  # 第三個數值
                    self.location_e4 = int(values_e[3])  # 第四個數值
                except ValueError:
                    print("輸入的不是有效的數值。")
            else:
                # 使用者取消或關閉對話框
                print("使用者取消輸入或關閉對話框")
        else:
            # 使用者按下取消或關閉對話框
            print("使用者取消輸入或關閉對話框")

        if self.location_w is not None:
            values_w = self.location_w.split()  # 用空白字符拆分字串
            if len(values_w) == 4:
                try:
                    self.location_w1 = int(values_w[0])  # 第一個數值
                    self.location_w2 = int(values_w[1])  # 第二個數值
                    self.location_w3 = int(values_w[2])
                    self.location_w4 = int(values_w[3])
                except ValueError:
                    print("輸入的不是有效的數值。")
            else:
                # 使用者取消或關閉對話框
                print("使用者取消輸入或關閉對話框")
        else:
            # 使用者按下取消或關閉對話框
            print("使用者取消輸入或關閉對話框")

        if self.location_s is not None:
            values_s = self.location_s.split()  # 用空白字符拆分字串
            if len(values_s) == 4:
                try:
                    self.location_s1 = int(values_s[0])  # 第一個數值
                    self.location_s2 = int(values_s[1])  # 第二個數值
                    self.location_s3 = int(values_s[2])
                    self.location_s4 = int(values_s[3])
                except ValueError:
                    print("輸入的不是有效的數值。")
            else:
                # 使用者取消或關閉對話框
                print("使用者取消輸入或關閉對話框")
        else:
            # 使用者按下取消或關閉對話框
            print("使用者取消輸入或關閉對話框")

        if self.location_n is not None:
            values_n = self.location_n.split()  # 用空白字符拆分字串
            if len(values_n) == 4:
                try:
                    self.location_n1 = int(values_n[0])  # 第一個數值
                    self.location_n2 = int(values_n[1])  # 第二個數值
                    self.location_n3 = int(values_n[2])
                    self.location_n4 = int(values_n[3])
                except ValueError:
                    print("輸入的不是有效的數值。")
            else:
                # 使用者取消或關閉對話框
                print("使用者取消輸入或關閉對話框")
        else:
            # 使用者按下取消或關閉對話框
            print("使用者取消輸入或關閉對話框")

    def open_file(self):
        file = askopenfile(mode='r', filetypes=[('Video Files', ["*.mp4"])])
        if file is not None:
            self.FileName = file.name
            print(self.FileName)
            self.IsOpenFile = True


            self.origin_n2 = {'x': 637, 'y': 438}
            self.target_n2 = {'x': 841, 'y': 427}
            self.list_n2 = remove_duplicates(get_points(self.origin_n2, self.target_n2))
            self.origin_s2 = {'x': 413, 'y': 445}
            self.target_s2 = {'x': 637, 'y': 448}
            self.list_s2 = remove_duplicates(get_points(self.origin_s2, self.target_s2))

            self.save_heatmap = self.FileName.split("/")[-1]
            self.save_heatmap = self.save_heatmap.split(".")[0]
            self.save_heatmap_img = f"./runs/heatmap/{self.save_heatmap}/img"
            self.save_heatmap_video = f"./runs/heatmap/{self.save_heatmap}/video"

            # 初始化結果
            self.e = 0
            self.s = 0
            self.n = 0
            self.w = 0
            self.label_Quantity_E.configure(text=str(self.e))
            self.label_Quantity_N.configure(text=str(self.n))
            self.label_Quantity_S.configure(text=str(self.s))
            self.label_Quantity_W.configure(text=str(self.w))
            self.label_Congestion_N.configure(text=str(""))
            self.label_Congestion_S.configure(text=str(""))
            self.label_Congestion_W.configure(text=str(""))
            self.label_Congestion_E.configure(text=str(""))


    def start1(self):
        if self.IsOpenFile:
            self.get_user_input()
            # 定義每一向的判斷位置
            self.origin_n = {'x': self.location_n1, 'y': self.location_n2}
            self.target_n = {'x': self.location_n3, 'y': self.location_n4}
            self.list_n = remove_duplicates(get_points(self.origin_n, self.target_n))
            self.origin_w = {'x': self.location_w1, 'y': self.location_w2}
            self.target_w = {'x': self.location_w3, 'y': self.location_w4}
            self.list_w = remove_duplicates(get_points(self.origin_w, self.target_w))
            self.origin_e = {'x': self.location_e1, 'y': self.location_e2}
            self.target_e = {'x': self.location_e3, 'y': self.location_e4}
            self.list_e = remove_duplicates(get_points(self.origin_e, self.target_e))
            self.origin_s = {'x': self.location_s1, 'y': self.location_s2}
            self.target_s = {'x': self.location_s3, 'y': self.location_s4}
            self.list_s = remove_duplicates(get_points(self.origin_s, self.target_s))
            self.CapVideo1(True)
        else:
           CTkMessagebox(title="Error", message="未選擇檔案！", icon="cancel")

    def CapVideo1(self, IsOpenVideo):
        if IsOpenVideo is True:
            self.heatmap_list = []
            self.cap = cv2.VideoCapture(self.FileName)
            self.label_Video.configure(text="")
            self.fps = self.cap.get(round(cv2.CAP_PROP_FPS,2))

            # 設定儲存位置
            directory_path = f"runs/detect"
            file_name = "result_"
            files = os.listdir(directory_path)

            self.num = 1
            while True:
                if f'{file_name}{self.num}.mp4' in files:
                    self.num += 1
                    continue
                else:
                    new_filename = f'{file_name}{self.num}.mp4'
                    break

            self.output_file = os.path.join(directory_path, new_filename)
            self.out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1240, 810))
            self.update1()


    def DrawResult1(self):
        data = [self.e, self.s, self.w, self.n]
        dir = ["E", "S", "W", "N"]
        fig = plt.figure(figsize=(3.5, 3.5))
        plt.subplot()
        plt.pie(data,
            radius=1,
            labels=dir,
            textprops={'weight':'bold', 'size':12},
            autopct='%.1f%%',
            wedgeprops={'linewidth':3, 'edgecolor':'w'})
        # plt.show()
        self.canvas1 = FigureCanvasTkAgg(fig, self.canvas)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")



    def update1(self):
        self.OpenFileBtn.configure(state='disabled')
        self.Start1Btn.configure(state='disabled')
        self.OpenPathBtn.configure(state='disabled')


        ret, frame = self.cap.read()
        if ret == True:

            frame = cv2.resize(frame, (1240, 810))  # (1240, 810)
            heatmap_frame = frame
            results = run_model(self.model, frame)
            track_ids = results.id.cpu().tolist()

            heatmap = np.zeros_like(heatmap_frame, dtype=np.float32) # 建立與圖像相同大小的陣列，用於存heatmap

            for track_id, detection in zip(track_ids, results.xyxy):
                xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)  # 繪製邊界框

                heatmap[ymin:ymax, xmin:xmax] += 1

                # 計算中心點座標
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)
                center = (center_x, center_y)

                if(track_id in self.track):
                    continue

                if (is_in_list(self.list_s, center)):
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    self.s += 1
                    self.label_Quantity_S.configure(text=str(self.s))
                    self.track.append(track_id)

                if (is_in_list(self.list_n, center)):
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    self.n += 1
                    self.label_Quantity_N.configure(text=str(self.n))
                    self.track.append(track_id)

                if (is_in_list(self.list_e, center)):
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    self.e += 1
                    self.label_Quantity_E.configure(text=str(self.e))
                    self.track.append(track_id)

                if (is_in_list(self.list_w, center)):
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    self.w += 1
                    self.label_Quantity_W.configure(text=str(self.w))
                    self.track.append(track_id)

               # print(">>>>>>>",self.s, self.n, self.e, self.w)
                #print(">>>>>>>", track_id, self.track)
               # if len(self.track) > 2:
                #    exit()

            heatmap = heatmap / np.max(heatmap)

            # 轉換heatmap
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # 疊加原始圖像和heatmap
            heatmap_result = cv2.addWeighted(heatmap_frame.astype("uint8"), 0.7, heatmap_color, 0.3, 0)
            self.heatmap_list.append(heatmap_result)

            # 標記FPS
            cv2.putText(frame, f'FPS: {round(self.fps,2)}',
                        (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)

            self.out.write(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            img = Image.fromarray(cv2image).resize((1080, 600))

            imgtks = ImageTk.PhotoImage(image=img)

            self.label_Video.imgtk = imgtks
            self.label_Video.configure(image=imgtks)
            self.label_Video.after(2, self.update1)

        else:
            sort = [self.s, self.n, self.e, self.w]
            level1 = "較高"
            level2 = "普通"
            level3 = "較低"
            sorted_sort = sorted(sort, reverse=True)
            if (sorted_sort[0] == self.n):
                self.label_Congestion_N.configure(text=level1)
                # label_n = level1
                self.label_Congestion_S.configure(text=level3)
                # label_s = level3
                self.label_Congestion_W.configure(text=level2)
                # label_w = level2
                self.label_Congestion_E.configure(text=level2)
                # label_e = level2
            elif (sorted_sort[0] == self.s):
                self.label_Congestion_N.configure(text=level3)
                # label_n = level3
                self.label_Congestion_S.configure(text=level1)
                # label_s = level1
                self.label_Congestion_W.configure(text=level2)
                # label_w = level2
                self.label_Congestion_E.configure(text=level2)
                # label_e = level2
            elif (sorted_sort[0] == self.w):
                self.label_Congestion_W.configure(text=level1)
                # label_w = level1
                self.label_Congestion_E.configure(text=level3)
                # label_e = level3
                self.label_Congestion_N.configure(text=level2)
                # label_n = level2
                self.label_Congestion_S.configure(text=level2)
                # label_s = level2
            else:
                self.label_Congestion_W.configure(text=level3)
                # label_w = level3
                self.label_Congestion_E.configure(text=level1)
                # label_e = level1
                self.label_Congestion_N.configure(text=level2)
                # label_n = level2
                self.label_Congestion_S.configure(text=level2)
                # label_s = level2

            # 設定車流量儲存位置
            self.output_txt_file = open('runs/txt/crossroads.txt', 'a')
            self.output_txt_file.write(f"第{self.num}次辨識 S: {self.s}, N: {self.n}, E: {self.e}, W: {self.w}\n")
            self.output_txt_file.close()

            self.cap.release()
            self.out.release()
            self.label_Video.configure(image="", text="請選擇影片", font=customtkinter.CTkFont(size=20, family="Arial"))
            self.DrawResult1()
            self.IsOpenFile = False
            self.OpenFileBtn.configure(state='normal')
            # self.OpenLiveBtn.configure(state='normal')
            self.StartBtn.configure(state='normal')
            self.Start1Btn.configure(state='normal')
            self.OpenPathBtn.configure(state='normal')

            save_heatmap_video_img(self.heatmap_list, self.save_heatmap_video, self.save_heatmap_img, self.fps)

    def start(self):
        if self.IsOpenFile:
            self.CapVideo(True)

        else:
           CTkMessagebox(title="Error", message="未選擇檔案！", icon="cancel")

    def CapVideo(self, IsOpenVideo):
        if IsOpenVideo is True:
            self.heatmap_list = []
            self.cap = cv2.VideoCapture(self.FileName)
            self.label_Video.configure(text="")
            # 標記FPS及數量
            self.fps = self.cap.get(round(cv2.CAP_PROP_FPS,2))

            # 設定儲存位置
            directory_path = f"runs/detect"
            file_name = "result_"
            files = os.listdir(directory_path)

            self.num = 1
            while True:
                if f'{file_name}{self.num}.mp4' in files:
                    self.num += 1
                    continue
                else:
                    new_filename = f'{file_name}{self.num}.mp4'
                    break

            self.output_file  = os.path.join(directory_path, new_filename)
            self.out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (1240, 810))
            self.update()

    def update(self):
        # Disabled Button
        self.OpenFileBtn.configure(state='disabled')
        # self.OpenLiveBtn.configure(state='disabled')
        self.StartBtn.configure(state='disabled')
        self.OpenPathBtn.configure(state='disabled')

        ret, frame = self.cap.read()
        if ret == True:

            frame = cv2.resize(frame, (1240, 810))  # (1240, 810)
            heatmap_frame = frame
            results = run_model(self.model, frame)
            detections = results.xyxy  # 獲取偵測到的物件和其邊界框座標

            track_ids = results.id.cpu().tolist()

            heatmap = np.zeros_like(heatmap_frame, dtype=np.float32)  # 建立與圖像相同大小的陣列，用於存heatmap

            for track_id, detection in zip(track_ids, results.xyxy):
                #class_idx = int(detection[-1].item())  # 獲取物件的類別標籤
                #if class_idx in [2, 7]:  # 只取汽車和卡車
                xmin, ymin, xmax, ymax = map(int, detection[:4].tolist())  # 解析邊界框座標
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)  # 繪製邊界框

                heatmap[ymin:ymax, xmin:xmax] += 1

                # 計算中心點座標
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)

                center=(center_x,center_y)

                if (track_id in self.track):
                    continue

                if (is_in_list(self.list_s2, center)):
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    self.s += 1
                    self.label_Quantity_S.configure(text=str(self.s))
                    self.track.append(track_id)

                if (is_in_list(self.list_n2, center)):
                    cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                    self.n += 1
                    self.label_Quantity_N.configure(text=str(self.n))
                    self.track.append(track_id)

            heatmap = heatmap / np.max(heatmap)

            # 轉換heatmap
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # 疊加原始圖像和heatmap
            heatmap_result = cv2.addWeighted(heatmap_frame.astype("uint8"), 0.7, heatmap_color, 0.3, 0)
            self.heatmap_list.append(heatmap_result)

            # 標記FPS
            cv2.putText(frame, f'FPS: {round(self.fps, 2)}',
                        (500, 40), cv2.FONT_HERSHEY_PLAIN, 3, WHITE, 3)

            self.out.write(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            img = Image.fromarray(cv2image).resize((1080, 600))

            imgtks = ImageTk.PhotoImage(image=img)

            self.label_Video.imgtk = imgtks
            self.label_Video.configure(image=imgtks)
            self.label_Video.after(2, self.update)


        else:
            if (self.s > self.n):
                self.label_Congestion_S.configure(text=str("較高"))
                self.label_Congestion_N.configure(text=str("較低"))
                self.label_Congestion_W.configure(text=str(""))
                self.label_Congestion_E.configure(text=str(""))
            else:
                self.label_Congestion_S.configure(text=str("較低"))
                self.label_Congestion_N.configure(text=str("較高"))
                self.label_Congestion_W.configure(text=str(""))
                self.label_Congestion_E.configure(text=str(""))
            # 設定儲存位置
            self.output_txt_file = open('runs/txt/two-way street.txt', 'a')
            self.output_txt_file.write(f"第{self.num}次辨識 S: {self.s}, N: {self.n}, E: {self.e}, W: {self.w}\n")
            self.output_txt_file.close()

            self.cap.release()
            self.out.release()
            self.label_Video.configure(image="", text="請選擇影片", font=customtkinter.CTkFont(size=20, family="Arial"))
            self.DrawResult()
            self.IsOpenFile = False
            self.OpenFileBtn.configure(state='normal')
            # self.OpenLiveBtn.configure(state='normal')
            self.StartBtn.configure(state='normal')
            self.Start1Btn.configure(state='normal')
            self.OpenPathBtn.configure(state='normal')

            save_heatmap_video_img(self.heatmap_list, self.save_heatmap_video, self.save_heatmap_img, self.fps)


    def DrawResult(self):
        data = [0, self.s, 0, self.n]
        dir = ["E", "S", "W", "N"]
        fig = plt.figure(figsize=(3.5, 3.5))
        plt.subplot()
        plt.pie(data,
            radius=1,
            labels=dir,
            textprops={'weight':'bold', 'size':12},
            autopct='%.1f%%',
            wedgeprops={'linewidth':3, 'edgecolor':'w'})
        # plt.show()
        self.canvas1 = FigureCanvasTkAgg(fig, self.canvas)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")

    def UpdateLabel(self, label, num):
        # This is change label text method
        # self.label_Direction.configure(text=str)
        pass

    def OpenFilePath(self):
        self.path = 'runs'
        os.startfile(self.path)

if __name__ == "__main__":
    app = main_window()
    app.mainloop()
