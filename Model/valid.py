from ultralytics import YOLO

def validate_yolov8(model_path, data_yaml, device='cuda'):
    # Load the model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=data_yaml, device=device)
    
    # Print results
    print("Validation Results:")
    print(results)
    
    return results

if __name__ == "__main__":
    model_path = "/notebooks/FINAL_MODEL/sightless_niv4/weights/best.pt"  # Change to your trained model path
    data_yaml = "/notebooks/Final_Dataset/dataset.yaml"     # Change to your dataset YAML path
    device = "cuda"             # Use 'cuda' if you have a GPU, otherwise 'cpu'
    
    validate_yolov8(model_path, data_yaml, device)
