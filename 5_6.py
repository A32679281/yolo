import numpy as np
import cv2
import matplotlib.pyplot as plt

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# 假設有一些原始座標點
points = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

# 假設這些是原始視角和目標視角的座標點
SOURCE = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)
TARGET = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)

# 初始化 ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# 進行座標轉換
transformed_points = view_transformer.transform_points(points)
red_point = np.array([[100, 50]], dtype=np.float32)

# 進行座標轉換
transformed_red_point = view_transformer.transform_points(red_point)

# 輸出轉換後的位置
print("Transformed Red Point:", transformed_red_point)
# 視覺化結果
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c='r', marker='o', label='Original Points')
plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='b', marker='^', label='Transformed Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original and Transformed Points')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()