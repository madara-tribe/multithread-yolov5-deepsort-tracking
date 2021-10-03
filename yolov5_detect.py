import sys
from pathlib import Path
import cv2, os
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import PIL.Image, PIL.ImageTk
from multiprocessing import Queue

from models.experimental import attempt_load
from option_parser import get_parser
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from Deepsort.utils.parser import get_config
from Deepsort.deep_sort.deep_sort import DeepSort


def preprocess_img(img, device, half=False):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img

def load_model(weights, device, half=False):
    global deepsort
    cfg = get_config()
    cfg.merge_from_file('Deepsort/configs/deep_sort.yaml')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    yolov5 = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        yolov5.half()  # to FP16
    return yolov5

def initialize():
    global device
    global half # use FP16 half-precision inference
    device=''
    half = False
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

def deepsort_detection(annotator, det, img, im0, names, s):
    if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(
            img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]

        # pass detections to deepsort
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
        
        # draw boxes for visualization
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                annotator.box_label(bboxes, label, color=colors(c, True))

    else:
        deepsort.increment_ages()
    return deepsort, annotator, s

@torch.no_grad()
def yolov5_detection(q:Queue, opt, save_vid=False, show_vid=False, tkinter_is=False):
    initialize()
    weights, source, imgsz, conf_thres, iou_thres = opt.weights, opt.source, opt.imgsz, opt.conf_thres, opt.iou_thres
    save_txt, classes, agnostic_nms, augment = opt.save_txt, opt.classes, opt.agnostic_nms, opt.augment
    nosave, exist_ok = opt.nosave, opt.exist_ok

    project = 'results'
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    # Directories
    save_dir = increment_path(project, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    
    yolov5 = load_model(weights, device, half=False)
    stride = int(yolov5.stride.max())  # model stride
    names = yolov5.module.names if hasattr(yolov5, 'module') else yolov5.names  # get class names
    
    # Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    bs = 1  # batch_size

    # Run inference
    if pt and device.type != 'cpu':
        yolov5(torch.zeros(1, 3, *imgsz).to(device).type_as(next(yolov5.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    t0 = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        # Inference
        t1 = time_sync()
        img = preprocess_img(img, device, half=half)
        pred = yolov5(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(project) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            deepsort, annotator, s = deepsort_detection(annotator, det, img, im0, names, s)
        
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            elif save_vid:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'

                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            elif tkinter_is:
                im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                im_rgb = cv2.resize(im_rgb, (1280, 720))
                q.put(im_rgb)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == "__main__":
    q = Queue()
    opt = get_parser()
    initialize()
    tkinter_is = False
    yolov5_detection(q, opt, save_vid=False, show_vid=True, tkinter_is=tkinter_is)
    
