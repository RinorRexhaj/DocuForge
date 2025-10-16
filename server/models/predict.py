import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


class EnhancedResNetModel(nn.Module):
    """
    Enhanced ResNet50 with attention mechanisms and optimized architecture
    for document forgery detection
    """
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(EnhancedResNetModel, self).__init__()

        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')

        # Get feature dimension from the last layer
        in_features = self.backbone.fc.in_features  # 2048 for ResNet50

        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()

        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_features, in_features // 16, kernel_size=1),
            nn.BatchNorm2d(in_features // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 16, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Channel Attention Module (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, in_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features),
            nn.Sigmoid()
        )

        # Multi-scale pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Enhanced classification head with residual connections
        self.classifier = nn.Sequential(
            # First block: 2048*2 -> 1024
            nn.Linear(in_features * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            # Second block: 1024 -> 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),

            # Third block: 512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),

            # Output layer
            nn.Linear(256, num_classes)
        )

        # Initialize weights with better initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize the weights of the classification head"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_probs=False):
        # Extract features from ResNet backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        batch_size, channels, height, width = x.size()

        # Spatial Attention
        spatial_att = self.spatial_attention(x)
        x_spatial = x * spatial_att

        # Channel Attention
        channel_att = self.channel_attention(x)
        channel_att = channel_att.view(batch_size, channels, 1, 1)
        x_channel = x * channel_att

        x_attended = x_spatial + x_channel

        # Multi-scale pooling
        avg_pool = self.global_avg_pool(x_attended).flatten(1)
        max_pool = self.global_max_pool(x_attended).flatten(1)
        features = torch.cat([avg_pool, max_pool], dim=1)

        # Classification head
        logits = self.classifier(features)

        # Optionally return probabilities instead of logits
        if return_probs:
            return torch.sigmoid(logits)

        return logits


def load_model(model_path='saved_models/best_model.pth', device=None):
    """
    Load the trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        device (torch.device, optional): Device to load the model on. If None, automatically selects.
    
    Returns:
        model: Loaded model in evaluation mode
        device: Device the model is loaded on
    """
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = EnhancedResNetModel(num_classes=1, dropout_rate=0.5)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    print(f"✅ Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"Best validation RECALL: {checkpoint.get('val_recall', 'N/A'):.4f} ({checkpoint.get('val_recall', 0)*100:.2f}%) ⭐")
    print(f"Best validation precision: {checkpoint.get('val_precision', 'N/A'):.4f}")
    print(f"Best validation F1-score: {checkpoint.get('val_f1', 'N/A'):.4f}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}\n")
    
    print(f"✅ Model loaded successfully from {model_path}")
    print(f"📍 Using device: {device}")
    
    return model, device


def get_transforms():
    """
    Get the image preprocessing transforms that match the training configuration.
    
    Returns:
        transforms.Compose: Composed transforms for preprocessing
    """
    IMG_SIZE = 224  # ResNet50 default input size
    
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict(image_path, model=None, model_path='saved_models/best_model.pth', 
            threshold=0.5, return_probability=False):
    """
    Predict whether a document image is authentic or forged.
    
    Args:
        image_path (str): Path to the image file
        model (nn.Module, optional): Pre-loaded model. If None, loads from model_path
        model_path (str): Path to the saved model (used if model is None)
        threshold (float): Decision threshold (default: 0.5)
        return_probability (bool): If True, returns probability. If False, returns label
    
    Returns:
        If return_probability is True:
            dict: {
                'prediction': str ('authentic' or 'forged'),
                'probability': float (probability of being forged),
                'confidence': float (confidence in the prediction)
            }
        If return_probability is False:
            str: 'authentic' or 'forged'
    """
    # Load model if not provided
    if model is None:
        model, device = load_model(model_path)
    else:
        device = next(model.parameters()).device
    
    # Load and preprocess image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probability = torch.sigmoid(logits).item()
    
    # Determine prediction
    prediction = 'forged' if probability > threshold else 'authentic'
    confidence = probability if probability > threshold else (1 - probability)
    
    if return_probability:
        return {
            'prediction': prediction,
            'probability': round(probability, 4),
            'confidence': round(confidence, 6)
        }
    else:
        return prediction


def predict_batch(image_paths, model=None, model_path='saved_models/best_model.pth', 
                  threshold=0.5, return_probability=False):
    """
    Predict multiple images in batch for efficiency.
    
    Args:
        image_paths (list): List of paths to image files
        model (nn.Module, optional): Pre-loaded model. If None, loads from model_path
        model_path (str): Path to the saved model (used if model is None)
        threshold (float): Decision threshold (default: 0.5)
        return_probability (bool): If True, returns probabilities. If False, returns labels
    
    Returns:
        list: List of predictions for each image
    """
    # Load model if not provided
    if model is None:
        model, device = load_model(model_path)
    else:
        device = next(model.parameters()).device
    
    # Prepare transforms
    transform = get_transforms()
    
    # Load and preprocess all images
    images = []
    valid_paths = []
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            images.append(image_tensor)
            valid_paths.append(img_path)
        else:
            print(f"⚠️ Warning: Image not found: {img_path}")
    
    if not images:
        return []
    
    # Stack images into batch
    batch = torch.stack(images).to(device)
    
    # Make predictions
    model.eval()
    results = []
    
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Handle single image case
        if len(images) == 1:
            probabilities = [probabilities.item()]
        else:
            probabilities = probabilities.tolist()
    
    # Process results
    for prob in probabilities:
        prediction = 'forged' if prob > threshold else 'authentic'
        confidence = prob if prob > threshold else (1 - prob)
        
        if return_probability:
            results.append({
                'prediction': prediction,
                'probability': prob,
                'confidence': confidence
            })
        else:
            results.append(prediction)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Document Forgery Detection - Prediction Module")
    print("=" * 60)
    
    # Example 1: Single image prediction
    # Uncomment and modify the path below to test
    # result = predict('path/to/your/image.jpg', return_probability=True)
    # print(f"\nPrediction: {result['prediction']}")
    # print(f"Probability (forged): {result['probability']:.4f}")
    # print(f"Confidence: {result['confidence']:.4f}")
    
    # Example 2: Batch prediction
    # image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    # results = predict_batch(image_paths, return_probability=True)
    # for i, result in enumerate(results):
    #     print(f"\nImage {i+1}: {result['prediction']} "
    #           f"(confidence: {result['confidence']:.2%})")
    
    print("\n📖 Usage Examples:")
    print("\n1. Simple prediction:")
    print("   from Predict import predict")
    print("   result = predict('document.jpg')")
    print("   print(result)  # 'authentic' or 'forged'")
    
    print("\n2. Detailed prediction:")
    print("   result = predict('document.jpg', return_probability=True)")
    print("   print(result['prediction'])  # 'authentic' or 'forged'")
    print("   print(result['probability'])  # probability of being forged")
    print("   print(result['confidence'])   # confidence in prediction")
    
    print("\n3. Batch prediction:")
    print("   from Predict import predict_batch")
    print("   images = ['doc1.jpg', 'doc2.jpg', 'doc3.jpg']")
    print("   results = predict_batch(images, return_probability=True)")
    
    print("\n4. Using a pre-loaded model:")
    print("   from Predict import load_model, predict")
    print("   model, device = load_model('saved_models/best_model.pth')")
    print("   result = predict('document.jpg', model=model)")
    
    print("\n" + "=" * 60)
