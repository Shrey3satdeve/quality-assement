from ultralytics import YOLO
import torch

def main():
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
    else:
        device = 'cpu'
        print("❌ GPU not detected. Training on CPU (Using mini-dataset for speed).")

    # Load a pre-trained YOLOv8 classification model
    model = YOLO('yolov8n-cls.pt')

    # Train the model using the mini dataset
    # We use a very small number of epochs for this "fast" run.
    print(f"🚀 Starting training on {device}...")
    results = model.train(
        data='mini_dataset', 
        epochs=2, 
        imgsz=64, 
        device=device,
        plots=True
    )
    
    print(f"Training finished! The model is saved at 'runs/classify/train/weights/best.pt'.")

if __name__ == '__main__':
    main()
