import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def classify_label(label):
    """Normalize helmet class labels across different YOLO model output formats.
    
    Handles variations like: helmet, no-helmet, No_Helmet, helmetless, without helmet,
    and also handles cases where names come as dict or list from different model versions.
    """
    # Handle dict format (Ultralytics some models) vs list format
    if isinstance(label, dict):
        # Get the first value from the dict
        for v in label.values():
            label = v
            break
    # Ensure label is a string
    if isinstance(label, (list, tuple)):
        label = label[0] if label else ""
    label = str(label).strip().lower()

    # Normalize to helmet/no-helmet buckets
    # Patterns that should be "helmet"
    helmet_patterns = ['helmet', 'with helmet', 'helmeted']
    # Patterns that should be "no-helmet"
    no_helmet_patterns = ['no-helmet', 'no_helmet', 'without helmet', 'no helmet', 'nohelmet', 'helmetless']

    if any(p in label for p in helmet_patterns):
        return 'helmet'
    if any(p in label for p in no_helmet_patterns):
        return 'no-helmet'

    # If label is just "no" or starts with "no-", treat as no-helmet
    if label == 'no' or label.startswith('no-'):
        return 'no-helmet'

    # Default: return the original label if we can't determine
    return label


def detect(save_img=False):
    """Real-time webcam inference with YOLO, FPS counter, and helmet/no-helmet tally."""
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # FPS and tally tracking variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_current = 0
    helmet_count = 0
    no_helmet_count = 0
    frame_drop_counter = 0
    max_frame_drops = 10  # Tolerate 10 frame drops before exiting webcam

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, d in enumerate(pred):  # detections per image
            if len(d) == 0:
                continue
            # Get the labels and coordinates
            d = d.cpu().numpy()

            # Count helmet/no-helmet from detections
            for *xyxy, conf, cls_id in d:
                cls_name = names[int(cls_id)].lower() if isinstance(names, list) else names[int(cls_id)]
                normalized = classify_label(cls_name)

                if normalized == 'helmet':
                    helmet_count += 1
                elif normalized == 'no-helmet':
                    no_helmet_count += 1

        # Calculate FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps_current = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()

        # Stream results
        for i, d in enumerate(pred):  # detections per image
            if len(d) == 0:
                continue

            # Get the labels and coordinates
            d = d.cpu().numpy()

            for *xyxy, conf, cls in reversed(d):
                # Convert coordinates
                xyxy = np.array(xyxy).astype(int)

                # Get class name and normalize
                cls_name = names[int(cls)].lower() if isinstance(names, list) else names[int(cls)]
                label = classify_label(cls_name)
                color = colors[int(cls)] if int(cls) < len(colors) else (0, 255, 0)

                # Draw bounding box
                plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)

        # Display FPS and tally in the corner
        tally_text = f"Helmet: {helmet_count} | No Helmet: {no_helmet_count}"
        fps_text = f"FPS: {fps_current:.1f}"
        # Draw background rectangles and text
        cv2.putText(im0, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(im0, tally_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Print time (inference + NMS)
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps_val = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps_val, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_val, (w, h))
                vid_writer.write(im0)

    # Print FPS and tally summary
    print(f'FPS: {fps_current:.1f} average')
    print(f'Helmet detections: {helmet_count}')
    print(f'No Helmet detections: {no_helmet_count}')

    if save_txt or save_img:
        s = f"\\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--model-type', type=str, default='yolov8', help='type of model: yolov7 or yolov8')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()