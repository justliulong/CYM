from ultralytics import CYM
from PIL import Image
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='ultralytics/weights/yolo11l-seg.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default='datasets/test/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_path', type=str, default='runs/segment/predict/', help='path to save results')
    # you can add other arguments here,for example, iou confidence threshold, etc.
    args = parser.parse_args()

    model = CYM(args.weights)

    results = model(source=args.source)

    for r in results:
        if r.masks is not None:
            print(len(r.masks))
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        # r.show()
        r.save(args.save_path, boxes=False, color_mode='instance')
