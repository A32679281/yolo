import cv2
import youtube_dl

# YouTube视频链接
url = "https://www.youtube.com/watch?v=JX96ej0-1hM"

# 设置youtube-dl选项
ydl_opts = {
    'format': 'best',
}

# 使用youtube-dl下载视频
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict.get("url", None)

# 打开视频流
cap = cv2.VideoCapture(video_url)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# 读取并显示视频帧，按q键退出
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('YouTube Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()
