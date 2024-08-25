import cv2
from tkinter import *

clicked_coordinates = {"x": None, "y": None}
x, y = 0, 0

def mouse_callback(event, X, Y, flags, param):
    global x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coordinates["x"] = X
        clicked_coordinates["y"] = Y
        x = X
        y = Y
        update_entry()

def update_entry():
    entry_x.delete(0, END)
    entry_y.delete(0, END)
    entry_x.insert(0, str(clicked_coordinates["x"]))
    entry_y.insert(0, str(clicked_coordinates["y"]))

def input_coordinates():
    top = Tk()
    top.title("输入坐標")
    window_width =top.winfo_screenwidth()
    window_height =top.winfo_screenheight()
    width = 200
    height = 100
    left_x = int((window_width - width)/2)       # 計算左上 x 座標
    top_y = int((window_height - height)/2) 
    top.geometry(f'{width}x{height}+{left_x}+{top_y}')
    global entry_x, entry_y

    label_x = Label(top, text="X :")
    label_x.grid(row=0, column=0)
    entry_x = Entry(top)
    entry_x.grid(row=0, column=1)

    label_y = Label(top, text="Y :")
    label_y.grid(row=1, column=0)
    entry_y = Entry(top)
    entry_y.grid(row=1, column=1)

    def on_closing():
        top.destroy()
        draw_line()

    top.protocol("WM_DELETE_WINDOW", on_closing)
    top.mainloop()

def draw_line():
    img = cv2.imread('nlnlsun.jpg')
    img = cv2.resize(img, (300, 300))
    cv2.line(img, (0, y), (300, y), color=(25, 33, 189), thickness=2, lineType=4)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('nlnlsun.jpg')
size = img.shape
w = size[1]
h = size[0]
img = cv2.resize(img, (300, 300))
cv2.imshow("img", img)
cv2.setMouseCallback("img", mouse_callback)
input_coordinates()
