"""
Inference & Single Image Predictor for Rock-Paper-Scissors
==========================================================
Usage:
    python predict.py --image path/to/image.png
    python predict.py --image Rock-Paper-Scissors/test/rock/testrock01-01.png
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = ['paper', 'rock', 'scissors']

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(128, len(CLASS_NAMES))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probabilities.argmax()

    result = {
        'prediction': CLASS_NAMES[pred_idx],
        'confidence': float(probabilities[pred_idx]),
        'all_probabilities': {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Predict Rock-Paper-Scissors Class for an Image")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    parser.add_argument('--model', type=str, default='training_output/best_rps_model.pth', help="Path to trained checkpoint")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device.type.upper()}")

    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found at {args.model}. Please train the model first.")
        return

    model = load_model(args.model, device)
    res = predict_image(args.image, model, device)

    print("\n" + "="*40)
    print(f" Image: {args.image}")
    print(f" Prediction: {res['prediction'].upper()} (Confidence: {res['confidence']*100:.2f}%)")
    print(" Class Probabilities:")
    for k, v in res['all_probabilities'].items():
        print(f"  • {k.capitalize():<10}: {v*100:6.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()
