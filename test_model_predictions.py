"""
Test Model Predictions - Check if model is working correctly
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import pickle

print('TensorFlow version:', tf.__version__)

# Import model from training script
from food_quality_detection import ViT, load_image

# Paths
MODEL_ASSETS_DIR = Path("model_assets")
CHECKPOINT_PATH = Path("vit_checkpoints/vit_food_quality.h5")

print("="*60)
print("MODEL PREDICTION TESTING")
print("="*60)

# Load class info
with open(MODEL_ASSETS_DIR / 'class_info.pkl', 'rb') as f:
    class_info = pickle.load(f)

with open(MODEL_ASSETS_DIR / 'model_config.pkl', 'rb') as f:
    model_config = pickle.load(f)

print(f"\nClasses: {class_info['num_classes']}")
print(f"Class names: {class_info['class_names'][:5]}...")

# Initialize model
print("\nInitializing model...")
model = ViT(
    image_size=model_config['image_size'],
    patch_size=model_config['patch_size'],
    num_classes=model_config['num_classes'],
    dim=model_config['dim'],
    depth=model_config['depth'],
    heads=model_config['heads'],
    mlp_dim=model_config['mlp_dim'],
    channels=model_config['channels']
)

# Build model
dummy_input = tf.zeros((1, 3, model_config['image_size'], model_config['image_size']))
_ = model(dummy_input, training=False)

# Load weights
print(f"Loading weights from {CHECKPOINT_PATH}...")
model.load_weights(str(CHECKPOINT_PATH))
print("✓ Model loaded!")

def test_image(image_path, expected_class=None):
    """Test a single image"""
    print("\n" + "="*60)
    print(f"Testing: {image_path}")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"❌ File not found!")
        return
    
    # Load and preprocess
    img_array = load_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model(img_array, training=False)
    probs = tf.nn.softmax(predictions[0]).numpy()
    
    # Get top 5 predictions
    top5_indices = np.argsort(probs)[-5:][::-1]
    
    print("\nTop 5 Predictions:")
    print("-" * 60)
    for i, idx in enumerate(top5_indices):
        class_name = class_info['class_names'][idx]
        confidence = probs[idx] * 100
        marker = "✓" if expected_class and class_name == expected_class else " "
        print(f"{marker} {i+1}. {class_name:20s} - {confidence:6.2f}%")
    
    # Predicted class
    predicted_idx = np.argmax(probs)
    predicted_class = class_info['class_names'][predicted_idx]
    confidence = probs[predicted_idx] * 100
    
    print("\n" + "-" * 60)
    print(f"Final Prediction: {predicted_class} ({confidence:.2f}%)")
    
    if expected_class:
        if predicted_class == expected_class:
            print("✅ CORRECT!")
        else:
            print(f"❌ WRONG! Expected: {expected_class}")
    
    return predicted_class, confidence

# Test with sample images from dataset
print("\n" + "="*60)
print("TESTING WITH TRAINING DATASET SAMPLES")
print("="*60)

dataset_dir = Path("dataset_food_quality/Test")

if dataset_dir.exists():
    # Test 1: Fresh Apple
    fresh_apple_dir = dataset_dir / "freshapples"
    if fresh_apple_dir.exists():
        sample = list(fresh_apple_dir.glob("*.jpg")) + list(fresh_apple_dir.glob("*.png"))
        if sample:
            test_image(str(sample[0]), "freshapples")
    
    # Test 2: Rotten Apple
    rotten_apple_dir = dataset_dir / "rottenapples"
    if rotten_apple_dir.exists():
        sample = list(rotten_apple_dir.glob("*.jpg")) + list(rotten_apple_dir.glob("*.png"))
        if sample:
            test_image(str(sample[0]), "rottenapples")
    
    # Test 3: Fresh Banana
    fresh_banana_dir = dataset_dir / "freshbanana"
    if fresh_banana_dir.exists():
        sample = list(fresh_banana_dir.glob("*.jpg")) + list(fresh_banana_dir.glob("*.png"))
        if sample:
            test_image(str(sample[0]), "freshbanana")
    
    # Test 4: Rotten Banana
    rotten_banana_dir = dataset_dir / "rottenbanana"
    if rotten_banana_dir.exists():
        sample = list(rotten_banana_dir.glob("*.jpg")) + list(rotten_banana_dir.glob("*.png"))
        if sample:
            test_image(str(sample[0]), "rottenbanana")

else:
    print(f"❌ Dataset directory not found: {dataset_dir}")

# Test uploaded image
print("\n" + "="*60)
print("TESTING UPLOADED IMAGE")
print("="*60)

uploads_dir = Path("uploads")
if uploads_dir.exists():
    uploaded_files = list(uploads_dir.glob("*.*"))
    if uploaded_files:
        print(f"Found {len(uploaded_files)} uploaded file(s)")
        for f in uploaded_files[:3]:  # Test max 3 files
            test_image(str(f))
    else:
        print("No uploaded files found")
else:
    print("Uploads directory not found")

print("\n" + "="*60)
print("TESTING COMPLETE!")
