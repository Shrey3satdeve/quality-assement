from ultralytics import YOLO
import os
from pathlib import Path

def main():
    model_path = 'runs/classify/train4/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Please ensure training is complete.")
        return

    print("Loading the trained model...")
    model = YOLO(model_path)

    # 1. Validate the model on the validation dataset to get overall metrics
    print("\n--- Validating Model ---")
    metrics = model.val(data='dataset')

    # metrics for classification typically contain top1 and top5 accuracies
    if hasattr(metrics, 'top1'):
        print(f"Top-1 Accuracy: {metrics.top1:.4f}")
        print(f"Top-5 Accuracy: {metrics.top5:.4f}")

    # 2. Run a prediction on a single random image from the validation set
    print("\n--- Testing on a Single Image ---")
    val_dir = Path('dataset/val')
    test_image = None
    
    # Grab the first image we can find in the validation directory
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images = list(val_dir.rglob(ext))
        if images:
            test_image = images[0]
            break

    if test_image:
        print(f"Testing image: {test_image}")
        results = model(str(test_image))
        
        for result in results:
            # Get the top predicted class index
            top1_index = result.probs.top1
            # Get the corresponding class name
            predicted_class = result.names[top1_index]
            # Get the confidence score
            confidence = result.probs.top1conf
            
            print(f"\nResult: The model predicted this image as '{predicted_class}' with {confidence * 100:.2f}% confidence.")
    else:
        print("No test images found in the dataset/val directory.")

if __name__ == '__main__':
    main()
