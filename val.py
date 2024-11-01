from ultralytics import CYM
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='ultralytics/weights/yolo11m-seg.pt', help='model.pt path')
    opt = parser.parse_args()

    model = CYM(opt.weight)
    model.val()
