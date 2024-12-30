# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse # 解析命令行参数
import os # 与操作系统进行交互，包含文件路径操作与解析
import platform
import sys # sys模块包含了与python解释器和它的环境有关的函数
from pathlib import Path # Path能够更加方便得对字符串路径进行处理

import torch

# __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径
FILE = Path(__file__).resolve()
# ROOT保存着当前项目的父目录
ROOT = FILE.parents[0]  # YOLOv5 root directory
# sys.path即当前python环境可以运行的路径,假如当前项目不在该路径中,就无法运行其中的模块,所以就需要加载路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT设置为相对路径
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 下面是 Ultralytics 官方定义的库，由于上一步已经把路径加载上了，所以现在可以导入，这个顺序不可以调换
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'weights/yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source) # 将输入的source转换为字符串，确保兼容性，方便后续处理
    # 判断是否保存推理后的图像。条件为 nosave 为 False 且 source 不以 '.txt' 结尾
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 判断source是不是视频/图像文件路径
    # Path()提取文件名。suffix：最后一个组件的文件扩展名。若source是"data/1.jpg"， 则Path(source).suffix是".jpg"，Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是否是链接
    # .lower()转化成小写.startswith('http://')返回True or Flas
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断source是否为网络流或数字摄像头
    # .isnumeric()是否是由数字组成，返回True or False
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # save_dir是保存运行结果的文件夹名，通过递增的方式来命名。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # 根据前面生成的路径创建文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device) # 选择计算设备（CUDA 或 CPU），如果系统支持 GPU，则优先使用 GPU
    # DetectMultiBackend定义在models.common模块中，是要加载的网络模型
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # 通过不同的输入源来设置不同的数据加载方式
    if webcam: # 使用摄像头作为输入
        view_img = check_imshow(warn=True) # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        # 加载输入数据流
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else: # 直接从source文件下读取图片
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # dt: 存储每一步骤的耗时 seen: 记录已经处理完了多少帧图片
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 遍历数据集中的每一帧数据
    for path, im, im0s, vid_cap, s in dataset:
        '''
         在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
          path：文件路径（即source）
          im: resize后的图片（经过了放缩操作）
          im0s: 原始图片
          vid_cap=none
          s： 图片的基本信息，比如路径，大小
        '''
        with dt[0]:
            # 将图像数据从 numpy 转换为 torch 张量，并加载到设备
            im = torch.from_numpy(im).to(model.device)
            # 根据 half 参数选择是否使用 FP16 半精度推理
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 归一化图像数据，将像素值从 [0, 255] 缩放到 [0.0, 1.0]
            im /= 255
            # 如果图像是三维的（即没有batch维度），则添加batch维度
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # 如果启用了特征图可视化，递增保存路径
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 执行前向传播，获取预测结果
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            # 非极大值抑制（NMS）处理
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # 遍历每张图片的预测结果
            seen += 1 # 统计已处理的图片数量
            # 处理视频流的输入
            if webcam: # 如果是通过摄像头输入（batch_size >= 1）
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else: # 单张图片或视频文件的输入
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # 图片/视频的保存路径
            p = Path(p)  # 将路径转换为Path对象
            save_path = str(save_dir / p.name)  # 保存的图片路径（如 img.jpg）
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # 保存的标签路径（如 img.txt）
            s += '%gx%g ' % im.shape[2:]  # 图片尺寸信息，添加到打印字符串中
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化比例，用于将坐标从img_size转换到原图尺寸
            imc = im0.copy() if save_crop else im0  # 如果需要裁剪保存目标框，则复制图片
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) # 初始化标注工具
            
            if len(det): # 如果有检测结果
                # 将预测框的坐标从 img_size 转换为原图尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印每个类别的检测数量
                for c in det[:, 5].unique(): # 遍历所有预测框的类别
                    n = (det[:, 5] == c).sum()  # 统计该类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 将类别和数量信息添加到打印字符串中

                # Write results
                for *xyxy, conf, cls in reversed(det): # 遍历每个检测框
                    if save_txt:  # 如果需要保存为文本文件
                        # 将坐标格式从 xyxy 转为 xywh，并归一化
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式，是否包含置信度
                        with open(f'{txt_path}.txt', 'a') as f: # 将标签写入文件
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # 如果需要保存图片或裁剪目标框
                        c = int(cls)  # 转换类别为整数
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True)) # 在图片上标注框和标签
                    if save_crop: # 如果需要裁剪目标框
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result() # 将标注后的结果提取出来
            # 显示检测结果
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) # 在窗口显示带检测框的图片
                cv2.waitKey(1)  # 等待1毫秒用于刷新窗口

            # 保存检测结果（带有检测框的图片或视频）
            if save_img:
                if dataset.mode == 'image': # 如果是单张图片
                    cv2.imwrite(save_path, im0) # 保存结果图片到指定路径
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # 如果保存路径改变，说明是新视频
                        vid_path[i] = save_path # 更新视频保存路径
                        if isinstance(vid_writer[i], cv2.VideoWriter): # 如果之前的视频写入器存在
                            vid_writer[i].release()  # 释放之前的视频写入器资源
                        if vid_cap:  # 如果是视频输入
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) # 获取视频的帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取视频的宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取视频的高度
                        else:  # 如果是流媒体输入
                            fps, w, h = 30, im0.shape[1], im0.shape[0] # 默认帧率为 30，宽高为图片尺寸
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        # 初始化视频写入器
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0) # 将检测结果帧写入视频

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每张图片的平均处理时间（单位：毫秒）
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    # 如果保存了结果（图片或文本标签），打印保存路径信息
    if save_txt or save_img: # 如果保存了标签，打印标签数量和保存路径
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # 如果设置了更新模型，执行优化更新
    if update:
        strip_optimizer(weights[0])  # 更新模型以移除优化器（例如，为模型文件瘦身）


# 用于解析命令行参数并返回这些参数的值，为模型进行推理时提供参数
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    # 解析命令行参数，并将结果存储在opt对象中
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1 # 如果imgsz长度为1，则将其值乘以2；否则保持不变
    print_args(vars(opt)) # 打印解析后的参数
    return opt


def main(opt):
    # 检查项目所需的依赖项，排除 'tensorboard' 和 'thop' 这两个库
    check_requirements(exclude=('tensorboard', 'thop'))
    # 使用命令行参数的字典形式调用run函数
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt() # 解析参数
    main(opt)
