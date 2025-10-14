"""
Example usage of the Predict.py module for document forgery detection
"""

from Predict import predict, predict_batch, load_model

# # Example 1: Simple single image prediction
# print("Example 1: Simple Prediction")
# print("-" * 50)
# try:
#     # Replace with your actual image path
#     image_path = "test/test/authentic/example.jpg"
#     result = predict(image_path)
#     print(f"Prediction: {result}")
# except Exception as e:
#     print(f"Error: {e}")

print("\n")

# # Example 2: Detailed prediction with probability
# print("Example 2: Detailed Prediction with Probability")
# print("-" * 50)
# try:
#     image_path = "dataset/test/forged/00138455_00138473_forged.png"
#     result = predict(image_path, return_probability=True)
#     print(f"Prediction: {result['prediction']}")
#     print(f"Probability (forged): {result['probability']:.4f}")
#     print(f"Confidence: {result['confidence']:.2%}")
# except Exception as e:
#     print(f"Error: {e}")

# print("\n")

# Example 3: Batch prediction
print("Example 3: Batch Prediction")
print("-" * 50)
try:
    image_paths = [
        "dataset/test/forged/00138455_00138473_forged.png",
        "dataset/test/authentic/2078615593_5635.png",
        "dataset/test/forged/2500003724_2500003976_forged.png"
    ]
    results = predict_batch(image_paths, return_probability=True)
    
    for i, result in enumerate(results):
        print(f"Image {i+1}: {result['prediction']} "
              f"(confidence: {result['confidence']:.2%})")
except Exception as e:
    print(f"Error: {e}")

# print("\n")

# Example 4: Using pre-loaded model for multiple predictions
# print("Example 4: Pre-loaded Model for Multiple Predictions")
# print("-" * 50)
# try:
#     # Load model once
#     model, device = load_model('saved_models/best_model.pth')
    
#     # Use the same model for multiple predictions
#     test_images = [
#         "dataset/test/forged/00138455_00138473_forged.png",
#         "dataset/test/forged/2078615593_5635_forged.png"
#     ]
    
#     for img_path in test_images:
#         result = predict(img_path, model=model, return_probability=True)
#         print(f"{img_path}: {result['prediction']} "
#               f"(prob: {result['probability']:.4f})")
# except Exception as e:
#     print(f"Error: {e}")
