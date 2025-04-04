import cv2
import torch
from torchvision import transforms
from PIL import Image
from framework.models import ann

# 定义与训练时一致的预处理步骤
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整大小为 32x32
    transforms.Grayscale(num_output_channels=3),  # 转为 3 通道的灰度图
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 从视频中提取帧
video_path = '../VideoDetection/mnist_video_cv2.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # OpenCV 读取的帧为 BGR，需要转换为 RGB 以符合 PIL 图像格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转为 PIL 格式以应用 torchvision.transforms
    frame_pil = Image.fromarray(frame)  # 转换为 PIL.Image
    frame_tensor = transform(frame_pil).unsqueeze(0)  # 转为张量并添加批次维度
    frames.append(frame_tensor)

cap.release()

# 将所有帧堆叠为一个批次
video_tensor = torch.cat(frames, dim=0)  # [num_frames, 3, 32, 32]
print(f"视频张量形状：{video_tensor.shape}")

# 初始化模型
model = ann(None,num_iterations=2)
checkpoint=torch.load(r'F:\DendriteandDD\Dendritic-computing-master\PyTorch\results\mnist\ann\mnist\checkpoint\best.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 切换为推理模式

# 模型推理
with torch.no_grad():
    outputs = model(video_tensor)  # [num_frames, num_classes]
    predictions = torch.argmax(outputs, dim=1)  # 获取每帧的预测类别

print(f"预测结果：{predictions}")
