import torch

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.conf = 0.25

image_path = 'D:/projects/soundspaces/collect_data/raw_data/episode_15/round_7/visual/image_96.png'

results = yolo_model(image_path)
detections = results.pandas().xyxy[0]
print(detections['name'].unique())