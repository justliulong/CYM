from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolo11l-seg-bcc2/weights/best.pt')

    model.val()