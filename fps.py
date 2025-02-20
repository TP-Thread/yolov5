import torch
import time
import cv2

# 加载 YOLOv5 本地模型
yolov5_path = '/home/vic/yolov5'
weight_path = 'weights/yolov5s-shufflenetv2/best.pt'
weight_path = 'weights/yolov5s-mobilenetv3/best.pt'
model = torch.hub.load(yolov5_path, 'custom', weight_path, source='local')

# 读取测试视频
cap = cv2.VideoCapture(4)
# 设置摄像头帧率和分辨率
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 获取相机帧率
camera_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"相机输入帧率: {camera_fps}")
# cap.release()

# 延迟10s再记录一分钟的平均帧率
cv2.waitKey(10000)

# 记录计算 FPS 开始时间（60 秒）
fps_start_time = time.time()
frame_count = 0

while cap.isOpened():
    # 检查是否超过 60 秒
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > 60:
        break  # 结束 FPS 计算

    ret, frame = cap.read()
    if not ret:
        break  # 读取失败

    # 进行目标检测
    t1 = time.time()
    results = model(frame)
    t2 = time.time()

    # 通过一帧图像计算目标检测的 FPS
    frame_count += 1
    fps = 1 / (t2 - t1)
    print(f"Frame {frame_count}: {fps:.2f} FPS")

    # 可视化
    img = results.render()[0]
    cv2.imshow('YOLOv5 Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 计算在相机帧率限制下，目标检测 60 秒内的平均 FPS
fps_end_time = time.time()
avg_fps = frame_count / (fps_end_time - fps_start_time)
print(f"平均帧率 (60 秒内): {avg_fps:.2f} FPS")

cap.release()
cv2.destroyAllWindows()
