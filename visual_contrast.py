import pandas as pd
import matplotlib.pyplot as plt
 
# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\s+', '_', regex=True)
 
# 读取csv文件
yolov5s_results = pd.read_csv("runs/train/yolov5s/results.csv")
yolov5_ghostnet_results = pd.read_csv("runs/train/yolov5s-ghostnet/results.csv")
yolov5_shufflenetv2_results = pd.read_csv("runs/train/yolov5s-shufflenetv2/results.csv")
yolov5_mobilenetv3_results = pd.read_csv("runs/train/yolov5s-mobilenetv3/results.csv")

# Clean column names
clean_column_names(yolov5s_results)
clean_column_names(yolov5_ghostnet_results)
clean_column_names(yolov5_shufflenetv2_results)
clean_column_names(yolov5_mobilenetv3_results)

# Plot precision curves
plt.figure()
# lable属性为曲线名称，自己可以定义
plt.plot(yolov5s_results['metrics/precision'], label="YOLOv5s")
plt.plot(yolov5_ghostnet_results['metrics/precision'], label="YOLOv5s-GhostNet")
plt.plot(yolov5_shufflenetv2_results['metrics/precision'], label="YOLOv5s-ShuffleNetv2")
plt.plot(yolov5_mobilenetv3_results['metrics/precision'], label="YOLOv5s-MobileNetv3")
plt.xlabel("epoch", fontsize=14)
plt.ylabel("precision", fontsize=14)
plt.legend()
# 图的标题
# plt.title("metrics/precision")
# 图片名称
plt.savefig("precision.png")

# Plot recall curves
plt.figure()
plt.plot(yolov5s_results['metrics/recall'], label="YOLOv5s")
plt.plot(yolov5_ghostnet_results['metrics/recall'], label="YOLOv5s-GhostNet")
plt.plot(yolov5_shufflenetv2_results['metrics/recall'], label="YOLOv5s-ShuffleNetv2")
plt.plot(yolov5_mobilenetv3_results['metrics/recall'], label="YOLOv5s-MobileNetv3")
plt.xlabel("epoch", fontsize=14)
plt.ylabel("recall", fontsize=14)
plt.legend()
plt.savefig("recall.png")

# Plot mAP@0.5 curves
plt.figure()
plt.plot(yolov5s_results['metrics/mAP_0.5'], label="YOLOv5s")
plt.plot(yolov5_ghostnet_results['metrics/mAP_0.5'], label="YOLOv5s-GhostNet")
plt.plot(yolov5_shufflenetv2_results['metrics/mAP_0.5'], label="YOLOv5s-ShuffleNetv2")
plt.plot(yolov5_mobilenetv3_results['metrics/mAP_0.5'], label="YOLOv5s-MobileNetv3")
plt.xlabel("epoch", fontsize=14)
plt.ylabel("mAP@0.5", fontsize=14)
plt.legend()
plt.savefig("mAP_0.5_comparison.png")

# Plot mAP@0.5:0.95 curves
plt.figure()
plt.plot(yolov5s_results['metrics/mAP_0.5:0.95'], label="YOLOv5s")
plt.plot(yolov5_ghostnet_results['metrics/mAP_0.5:0.95'], label="YOLOv5s-GhostNet")
plt.plot(yolov5_shufflenetv2_results['metrics/mAP_0.5:0.95'], label="YOLOv5s-ShuffleNetv2")
plt.plot(yolov5_mobilenetv3_results['metrics/mAP_0.5:0.95'], label="YOLOv5s-MobileNetv3")
plt.xlabel("epoch", fontsize=14)
plt.ylabel("mAP@0.5:0.95", fontsize=14)
plt.legend()
plt.savefig("mAP_0.5_0.95_comparison.png")
