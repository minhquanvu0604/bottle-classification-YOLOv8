from ultralytics import YOLO

from trainer import CustomTrainer

model = YOLO('yolov8n-cls.pt', task='classify', verbose=True)

custom_trainer = CustomTrainer()

model.train(custom_trainer)
