from ultralytics import CYM, YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/cym/cym.yaml", help='model.yaml path')
    parser.add_argument("--epochs", type=int, default=100, help='number of epochs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/bcc.yaml", help='data.yaml path')
    parser.add_argument("--device", type=str, default="0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--batch", type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument("--weights",
                        type=str,
                        default="",
                        help='initial weights path,if you want to use the pretrained weights, you need to set the path')
    parser.add_argument("--workers", type=int, default=1, help='number of dataloader workers')
    parser.add_argument("--project", type=str, default="runs/train", help='save to project/name')
    parser.add_argument("--name", type=str, default="exp", help='save to project/name')
    args = parser.parse_args()

    model = CYM(model=args.model)
    # model = YOLO("ultralytics/cfg/models/11/yolo11m-seg.yaml")
    # model.load("./ultralytics/weights/yolo11m-seg.pt") # use the pretrained weights, if you want not to use it can comments it

    result = model.train(data=args.data,
                         epochs=args.epochs,
                         imgsz=args.imgsz,
                         batch=args.batch,
                         workers=args.workers,
                         project=args.project,
                         name=args.name)
