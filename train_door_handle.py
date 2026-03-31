'''import os
# Set the fallback so that operations not implemented on MPS (like NMS) are run on CPU.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO

# Print the current working directory for verification
print("Current Working Directory:", os.getcwd())

# Path to your dataset YAML file for door-handle detection.
# Make sure your data.yaml is updated and your folder structure matches.
data_yaml_path = "/Users/tripathd/Downloads/Manual Library/Projects/Accesbility_Device/Just_Handle_door/data.yaml"

# Initialize a YOLOv8 model.
# (We use yolov8n.pt as a starting point, as used for the stairs model.)
model = YOLO("yolov8n.pt")

# Train the model.
# The following parameters mimic your stairs model training:
# - epochs: 50 (adjust if needed; you mentioned that 50 epochs takes too long, so you might try 20–25)
# - batch: 16
# - imgsz: 640
# - device: "mps" (using the MPS backend on your M4 Max)
# - amp: True to enable automatic mixed precision training.
model.train(
    data=data_yaml_path,
    epochs=5,            # Set to 50 for full training; reduce this number if you want a shorter training run
    batch=16,
    imgsz=640,
    device="mps",         # Use MPS on Apple Silicon; the fallback variable ensures NMS issues are resolved
    amp=True,             # Enable automatic mixed precision for speed
    workers=8,            # Adjust the number of data loader workers as needed
    verbose=True
    
)
'''



import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from ultralytics import YOLO
print("Current Working Directory:", os.getcwd())
data_yaml_path = "/path/to/data.yaml"
model = YOLO("runs/detect/train23/weights/last.pt")  # Resuming from checkpoint
model.train(data=data_yaml_path, epochs=20, resume=True, device="mps")
