import cv2

# 加載圖像
img = cv2.imread('nlnlsun.jpg')

# 檢查圖像是否成功加載
if img is None:
    print("Error: Unable to load image.")
else:
    # 檢查圖像大小
    if img.shape[0] > 0 and img.shape[1] > 0:
        # 顯示圖像
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Image dimensions are invalid.")