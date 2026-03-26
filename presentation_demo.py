import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the trained model
# Change 'train4' to your best model directory if different
MODEL_PATH = 'runs/classify/train4/weights/best.pt' 
model = YOLO(MODEL_PATH)

def predict_and_show(image_path):
    # Run the image through your trained YOLO model
    results = model(image_path)
    
    # Get the name of exactly what the model predicts
    top1_index = results[0].probs.top1
    predicted_class = results[0].names[top1_index]
    confidence = results[0].probs.top1conf * 100

    # Load image for display using OpenCV
    img = cv2.imread(image_path)
    
    # Add text overlay to the original image
    text = f"{predicted_class} ({confidence:.1f}%)"
    
    # Determine text color based on good or bad (Green for Good, Red for Bad)
    if "Good" in predicted_class:
        color = (0, 255, 0) # Green 
    elif "Bad" in predicted_class:
        color = (0, 0, 255) # Red
    else:
        color = (255, 255, 0) # Cyan for 'mixed' or general

    cv2.putText(img, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Resize image to fit comfortably on screen for presentation
    img_resized = cv2.resize(img, (600, 600))
    
    # Show the resulting image
    cv2.imshow("Fruit Quality Assessment Tool - Presentation Mode", img_resized)
    cv2.waitKey(0) # Press any key to close the window
    cv2.destroyAllWindows()

def open_file_dialog():
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    print("Please select an image to test...")
    
    file_path = filedialog.askopenfilename(
        title="Select a fruit image to assess",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if file_path:
        print(f"Assessing quality for: {file_path}")
        predict_and_show(file_path)
    else:
        print("No file selected.")

if __name__ == '__main__':
    open_file_dialog()
