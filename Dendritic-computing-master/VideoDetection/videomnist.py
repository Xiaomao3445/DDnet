import torch
import torchvision
import torchvision.transforms as transforms
import random
import cv2
import numpy as np

# 定义 transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为32x32
    transforms.Grayscale(num_output_channels=3),  # 转为3通道的灰度图
    transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 加载 MNIST 数据集
mnist_dataset = torchvision.datasets.MNIST(root='F:/datasets/MNIST', train=False, download=True, transform=transform)

# 随机选择 10 张图片
num_frames = 20
selected_indices = random.sample(range(len(mnist_dataset)), num_frames)
images = torch.stack([mnist_dataset[i][0] for i in selected_indices])  # 获取图片张量

# 反标准化：恢复到 [0, 1] 范围
images_for_video = images * 0.3081 + 0.1307  # 反标准化
images_for_video = images_for_video.clip(0, 1)  # 限制范围在 [0, 1]

# 将张量转换为 OpenCV 需要的格式
# 1. 转换为 [0, 255] 范围
images_for_video = (images_for_video * 255).byte()  # 将像素值恢复到 [0, 255] 范围
# 2. 转为 BGR 格式，因为 OpenCV 使用 BGR 格式
images_for_video = images_for_video.permute(0, 2, 3, 1)  # 变为 [num_frames, height, width, 3] 格式
images_for_video = images_for_video.numpy()  # 转为 numpy 数组

# 创建 VideoWriter 对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式编码
fps = 2  # 设置视频帧率
frame_height, frame_width = images_for_video[0].shape[:2]  # 获取图像尺寸
video_writer = cv2.VideoWriter('mnist_video_cv2(20).mp4', fourcc, fps, (frame_width, frame_height))

# 写入视频帧
for img in images_for_video:
    # OpenCV 使用 BGR 格式，MNIST 是灰度图像，所以直接将灰度图像的 R, G, B 通道设置为一样
    #bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 转为 BGR 格式
    video_writer.write(img)  # 写入视频帧

# 释放资源
video_writer.release()
cv2.destroyAllWindows()

print("视频已保存为 mnist_video_cv2.mp4")
