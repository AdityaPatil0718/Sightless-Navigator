from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the YOLOv8n model
    model = YOLO("/notebooks/FINAL_MODEL/sightless_niv2/weights/last.pt").to(device)

    # Training configuration
    model.train(
        data="/notebooks/Final_Dataset/dataset.yaml",    # Path to dataset configuration
        epochs=70,              # Number of training epochs
        imgsz=640,              # Input image size
        batch=16,               # Batch size
        device="cuda",          # Automatically select device
        optimizer="AdamW",     # Optimizer (AdamW for better performance)
        lr0=0.0001,              # Initial learning rate
        weight_decay=0.0005,    # Weight decay for regularization
        save=True,              # Save model checkpoints
        project="FINAL_MODEL",       # Directory to save training results
        name="sightless_niv",  # Experiment name
        workers=4,              # Number of data loading workers
        patience=10,            # Early stopping patience
        val=True,               # Enable validation after each epoch
        augment=True            # Enable data augmentation
    )
