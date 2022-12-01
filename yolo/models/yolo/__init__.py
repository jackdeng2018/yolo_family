from yolo.models.yolo.yolov1 import YOLOv1


# build YOLO detector
def build_model(args, cfg, device, num_classes=80, trainable=False):
    if args.model == 'yolov1':
        print('Build YOLOv1 ...')
        model = YOLOv1(cfg=cfg,
                       device=device,
                       img_size=args.img_size,
                       num_classes=num_classes,
                       trainable=trainable,
                       conf_thresh=args.conf_thresh,
                       nms_thresh=args.nms_thresh,
                       center_sample=args.center_sample)

    return model