from ultralytics import YOLO
from PIL import Image

# model = YOLO("runs/train/yolo11l-seg-bcc2/weights/best.pt")
model = YOLO('ultralytics/weights/yolo11l-seg.pt')

path = "datasets/test/val/01.jpg"
results = model(path)

for r in results:
    if r.masks is not None:
        print(len(r.masks))
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])
    # r.show()
    r.save("runs/segment/predict/test.jpg", boxes=False, color_mode='instance')
