# import torch
import cv2

# # Model
# model = torch.hub.load("./", "yolov5s", source="local")

# model.conf = 0.25  # NMS confidence threshold
#     #   iou = 0.45  # NMS IoU threshold
#     #   agnostic = False  # NMS class-agnostic
#     #   multi_label = False  # NMS multiple labels per box
#     #   classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#     #   max_det = 1000  # maximum number of detections per image
#     #   amp = False  # Automatic Mixed Precision (AMP) inference

# #Images
# img="./data/images/zidane.jpg"

# # Inference
# results = model(img)
# # results = model(im, size=320)  # custom inference size

# # Results
# results.show()

# 打开摄像头 默认为0，cv2.CAP_V4L2 是Video for Linux 2的一个常量，用于指定使用V4L2驱动程序进行视频捕获
cap = cv2.VideoCapture(4, cv2.CAP_V4L2)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second (FPS): {fps}")
# 获取视频的宽度和高度
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Video dimensions: {width} x {height}")

while True:
    # 每次读取一帧摄像头或者视频
    ret,frame = cap.read()
    # 将一帧frame显示出来，第一个参数为窗口名
    cv2.imshow('frame',frame)
    # 每次等待1ms 当esc按键被按下时退出显示，ESC按键对应的键值为27
    if(cv2.waitKey(1)&0xff) == 27:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
