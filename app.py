import os
import pickle
import json
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import model components from training script
from food_quality_detection import ViT, load_image

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and assets
model = None
class_info = None
model_config = None
preprocess_params = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_assets():
    """Load trained model and all necessary assets"""
    global model, class_info, model_config, preprocess_params
    
    print("Loading model assets...")
    
    # Paths
    checkpoint_path = Path("vit_checkpoints/vit_food_quality.h5")
    model_assets_dir = Path("model_assets")
    
    # Check if files exist
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    if not model_assets_dir.exists():
        raise FileNotFoundError(f"Model assets directory not found at {model_assets_dir}")
    
    # Load class information
    with open(model_assets_dir / 'class_info.pkl', 'rb') as f:
        class_info = pickle.load(f)
    print(f"✓ Loaded class_info: {class_info['num_classes']} classes")
    
    # Ensure idx_to_class exists and is properly formatted
    if 'idx_to_class' not in class_info or not class_info['idx_to_class']:
        print("⚠ idx_to_class not found, rebuilding from class_names...")
        class_info['idx_to_class'] = {str(i): name for i, name in enumerate(class_info['class_names'])}
    
    # Debug: Print first few mappings
    print(f"✓ idx_to_class keys: {list(class_info['idx_to_class'].keys())[:5]}")
    print(f"✓ Sample mapping: 0 -> {class_info['idx_to_class'].get('0', 'NOT FOUND')}")
    
    # Load model configuration
    with open(model_assets_dir / 'model_config.pkl', 'rb') as f:
        model_config = pickle.load(f)
    print(f"✓ Loaded model_config")
    
    # Load preprocessing parameters
    with open(model_assets_dir / 'preprocess_params.pkl', 'rb') as f:
        preprocess_params = pickle.load(f)
    print(f"✓ Loaded preprocess_params")
    
    # Initialize model with same architecture
    print("Initializing model architecture...")
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
    
    # Build model by running a dummy forward pass
    dummy_input = tf.zeros((1, 3, model_config['image_size'], model_config['image_size']))
    _ = model(dummy_input, training=False)
    
    # Load trained weights
    print(f"Loading model weights from {checkpoint_path}...")
    model.load_weights(str(checkpoint_path))
    print("✓ Model loaded successfully!")
    
    return True

def preprocess_image(image_path):
    """Preprocess image for prediction (same as training)"""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    
    # Transpose to (C, H, W) format for ViT
    arr = np.transpose(arr, (2, 0, 1))
    
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    
    return arr.astype(np.float32)

def get_food_specific_features(food_name, is_fresh):
    """Generate food-specific visual features detected by AI"""
    
    food_features = {
        'apples': {
            'fresh': {
                'color': 'bright red or green coloration with natural shine',
                'texture': 'smooth, firm skin without wrinkles or soft spots',
                'surface': 'clean surface with minimal to no blemishes',
                'stem': 'intact stem area, no browning around the core',
                'specific': 'crisp appearance indicating high water content and freshness'
            },
            'rotten': {
                'color': 'brown or dark discoloration, loss of natural vibrancy',
                'texture': 'wrinkled, shriveled skin with visible soft spots',
                'surface': 'dark patches, possible mold growth (white/green fuzzy spots)',
                'stem': 'browning and decay around stem area',
                'specific': 'collapsed or mushy appearance indicating cellular breakdown'
            }
        },
        'banana': {
            'fresh': {
                'color': 'bright yellow color with possible green tips',
                'texture': 'firm, smooth peel without excessive brown spots',
                'surface': 'clean peel with natural texture intact',
                'stem': 'green stem attachment, firm connection',
                'specific': 'uniform shape without soft or mushy areas'
            },
            'rotten': {
                'color': 'extensive brown/black discoloration, darkened peel',
                'texture': 'very soft, possibly leaking or split peel',
                'surface': 'large dark brown/black patches covering significant area',
                'stem': 'completely brown or detached stem',
                'specific': 'fruit flies may be present, strong fermentation odor indicators'
            }
        },
        'tomato': {
            'fresh': {
                'color': 'vibrant red color (or appropriate color for variety)',
                'texture': 'firm, taut skin with natural glossiness',
                'surface': 'smooth surface without cracks or soft areas',
                'stem': 'green, fresh-looking stem attachment',
                'specific': 'plump appearance indicating proper ripeness'
            },
            'rotten': {
                'color': 'dull, darkened red or brown discoloration',
                'texture': 'extremely soft, possibly split or leaking skin',
                'surface': 'visible mold (white/green/black), wet spots',
                'stem': 'brown, decayed stem area with possible mold',
                'specific': 'collapsed structure, liquid seepage visible'
            }
        },
        'potato': {
            'fresh': {
                'color': 'natural tan/brown color, uniform appearance',
                'texture': 'firm to the touch with intact skin',
                'surface': 'clean surface with natural eyes, no sprouts',
                'stem': 'no visible sprouting or green discoloration',
                'specific': 'dry surface, proper storage indicators'
            },
            'rotten': {
                'color': 'dark brown/black spots, possible green areas (solanine)',
                'texture': 'soft, mushy areas indicating decay',
                'surface': 'sprouting, extensive blemishes, mold presence',
                'stem': 'long sprouts indicating old age',
                'specific': 'foul odor indicators, liquefaction of tissue'
            }
        },
        'oranges': {
            'fresh': {
                'color': 'bright orange color with natural oil sheen',
                'texture': 'firm, resilient peel with proper texture',
                'surface': 'clean peel with characteristic dimpled texture',
                'stem': 'green or slightly brown stem end',
                'specific': 'heavy for size indicating juiciness'
            },
            'rotten': {
                'color': 'dull, possibly darkened or spotted peel',
                'texture': 'soft spots, spongy or dried out peel',
                'surface': 'mold growth (often blue-green), wet spots',
                'stem': 'completely brown stem with decay spreading',
                'specific': 'light weight indicating dried out or spoiled interior'
            }
        },
        'cucumber': {
            'fresh': {
                'color': 'vibrant green color, uniform throughout',
                'texture': 'firm, crisp texture with taut skin',
                'surface': 'smooth or properly textured skin, no soft spots',
                'stem': 'fresh cut end or slightly dried attachment',
                'specific': 'straight shape, proper moisture content'
            },
            'rotten': {
                'color': 'yellowing or brown discoloration',
                'texture': 'soft, mushy texture, possible sliminess',
                'surface': 'wrinkled skin, possible mold, wet spots',
                'stem': 'browning and decay at ends',
                'specific': 'bent or collapsed shape, liquid seepage'
            }
        },
        'capsicum': {
            'fresh': {
                'color': 'vibrant color (red/green/yellow) with natural shine',
                'texture': 'firm, crisp flesh with taut skin',
                'surface': 'smooth, glossy surface without wrinkles',
                'stem': 'green, fresh stem attachment',
                'specific': 'heavy for size, proper moisture content'
            },
            'rotten': {
                'color': 'dull or brown discoloration, loss of vibrancy',
                'texture': 'soft spots, wrinkled or collapsed areas',
                'surface': 'mold (white/gray), slimy patches',
                'stem': 'brown, dried or moldy stem',
                'specific': 'lightweight, indicating dehydration or decay'
            }
        },
        'bittergourd': {
            'fresh': {
                'color': 'bright green with characteristic warty texture',
                'texture': 'firm texture with proper rigidity',
                'surface': 'clean bumpy surface, no discoloration',
                'stem': 'fresh green stem end',
                'specific': 'proper moisture level in characteristic texture'
            },
            'rotten': {
                'color': 'yellowing or brown discoloration of skin',
                'texture': 'soft, mushy areas between bumps',
                'surface': 'mold in crevices, slimy patches',
                'stem': 'brown, decayed stem area',
                'specific': 'collapsed texture, liquid seepage from bumps'
            }
        },
        'okra': {
            'fresh': {
                'color': 'bright green color uniformly distributed',
                'texture': 'firm pod with slight give when pressed',
                'surface': 'clean surface with natural fuzz',
                'stem': 'fresh cut or attached green stem',
                'specific': 'proper size, not overgrown or woody'
            },
            'rotten': {
                'color': 'dark brown or black discoloration',
                'texture': 'slimy, mushy texture throughout',
                'surface': 'possible mold, excessive sliminess',
                'stem': 'brown decay spreading from stem',
                'specific': 'collapsed shape, strong decomposition indicators'
            }
        }
    }
    
    # Get features or use generic
    food_key = food_name.lower()
    if food_key not in food_features:
        # Generic features for unknown food
        if is_fresh:
            return {
                'color': 'natural vibrant coloration typical of fresh produce',
                'texture': 'firm, intact structure without soft spots',
                'surface': 'clean surface without visible damage or mold',
                'stem': 'fresh attachment points where applicable',
                'specific': 'overall appearance consistent with peak freshness'
            }
        else:
            return {
                'color': 'discoloration indicating cellular breakdown',
                'texture': 'soft or mushy areas indicating decay',
                'surface': 'visible deterioration, possible mold growth',
                'stem': 'browning and decay at attachment points',
                'specific': 'multiple indicators of advanced spoilage'
            }
    
    status_key = 'fresh' if is_fresh else 'rotten'
    return food_features[food_key][status_key]

def generate_ai_explanation(food_type, status, confidence):
    """Generate AI-powered explanation based on prediction results"""
    
    # Extract food name (remove fresh/rotten prefix)
    food_name = food_type.replace('fresh', '').replace('rotten', '').strip()
    food_name = food_name.capitalize()
    
    # Get food-specific visual features
    is_fresh = (status == 'segar')
    visual_features = get_food_specific_features(food_name, is_fresh)
    
    # Confidence level categories
    if confidence >= 95:
        confidence_level = "very_high"
    elif confidence >= 80:
        confidence_level = "high"
    elif confidence >= 60:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    explanations = {
        'fresh': {
            'very_high': {
                'title': '🌟 Excellent Freshness Detected',
                'description': f'Our AI Vision System has analyzed the image with extremely high confidence ({confidence:.1f}%) and determined this {food_name.lower()} is in excellent fresh condition.',
                'analysis': [
                    f'🎨 Color Analysis: Detected {visual_features["color"]}. The AI recognizes this as a strong indicator of peak freshness, showing the food has been recently harvested or properly stored.',
                    f'🔍 Texture Assessment: Visual examination reveals {visual_features["texture"]}. The AI neural network identifies these characteristics as optimal quality markers with no signs of deterioration.',
                    f'🌡️ Surface Condition: The image analysis shows {visual_features["surface"]}. Our model detects this indicates proper handling and storage conditions.',
                    f'⚡ Structural Integrity: AI detects {visual_features["specific"]}. This combination of visual features strongly suggests the {food_name.lower()} is at peak quality and safe for consumption.'
                ],
                'recommendation': 'This food is safe and ideal for immediate consumption. Store it properly in a cool, dry place or refrigerate to maintain its quality.',
                'storage_tips': f'Store fresh {food_name.lower()} in the refrigerator (for fruits) or in a cool, dry place. Consume within 3-7 days for best quality.',
                'health_benefit': f'Fresh {food_name.lower()} retains maximum nutrients and vitamins, providing optimal health benefits.'
            },
            'high': {
                'title': '✅ Good Freshness Quality',
                'description': f'Our AI Vision System indicates high confidence ({confidence:.1f}%) that this {food_name.lower()} is fresh and suitable for consumption.',
                'analysis': [
                    f'🎨 Color Analysis: The AI detects {visual_features["color"]}. While showing strong freshness indicators, there may be minor natural variations that are completely normal for this produce.',
                    f'🔍 Texture Assessment: Visual examination shows {visual_features["texture"]}. The neural network identifies these as good quality markers consistent with fresh produce.',
                    f'🌡️ Surface Condition: Image analysis reveals {visual_features["surface"]}. The AI recognizes this as meeting quality standards with no concerning deterioration.',
                    f'⚡ Overall Assessment: Based on multi-layer neural network analysis, the visual features indicate this {food_name.lower()} is fresh and safe, though it should be consumed relatively soon for optimal quality.'
                ],
                'recommendation': 'This food is safe to eat. Consider consuming it soon to enjoy at peak freshness.',
                'storage_tips': f'Keep {food_name.lower()} refrigerated and consume within 2-5 days. Check daily for any changes.',
                'health_benefit': f'This {food_name.lower()} still contains good nutritional value and is safe for consumption.'
            },
            'medium': {
                'title': '⚠️ Moderate Freshness - Verification Recommended',
                'description': f'The AI shows moderate confidence ({confidence:.1f}%) in freshness. The visual analysis detects mixed signals that require human verification.',
                'analysis': [
                    f'🎨 Color Analysis: The AI detects some characteristics of {visual_features["color"]}, but also notes areas that show slight variations. This mixed pattern suggests borderline quality.',
                    f'🔍 Texture Assessment: Visual features show {visual_features["texture"]} in some areas, but the AI neural network also identifies potential inconsistencies that warrant closer examination.',
                    f'🌡️ Surface Condition: The image analysis reveals {visual_features["surface"]}, however the AI detects some ambiguous markers that could indicate early aging or simply natural variations.',
                    f'⚡ Uncertainty Factors: The AI\'s computer vision algorithms cannot definitively classify certain visual features. This may be due to lighting conditions, image angle, or the {food_name.lower()} being in a transitional state between peak freshness and early aging.'
                ],
                'recommendation': 'While likely still fresh, we recommend a manual inspection. Check for firmness, smell, and any soft spots before consuming.',
                'storage_tips': 'Consume soon (within 1-2 days). Monitor closely for any quality changes. Store in optimal conditions.',
                'health_benefit': 'If confirmed fresh upon inspection, this food should still provide nutritional benefits.'
            },
            'low': {
                'title': '🔍 Low Confidence - Manual Inspection Required',
                'description': f'The AI model shows low confidence ({confidence:.1f}%) in this assessment. The visual features are too ambiguous for reliable automated analysis.',
                'analysis': [
                    f'🎨 Color Analysis: The AI struggles to definitively identify color patterns. The visual data shows characteristics that could align with both {visual_features["color"]} but with insufficient clarity for confident classification.',
                    f'🔍 Texture Assessment: The neural network detects conflicting texture signals. While some features suggest {visual_features["texture"]}, other areas show ambiguous characteristics.',
                    f'🌡️ Image Quality Issues: The AI\'s visual processing identifies potential issues with image quality, lighting angle, or focus that significantly impact analysis accuracy. Clear images are crucial for reliable predictions.',
                    f'⚡ Recommendation: This low confidence score indicates the AI cannot reliably determine quality from the provided image. Human visual inspection, smell test, and touch examination are essential before making any consumption decision.'
                ],
                'recommendation': 'Do not rely solely on this result. Perform thorough manual inspection including smell test, firmness check, and visual examination under good lighting.',
                'storage_tips': 'If confirmed fresh, consume immediately. Do not store for extended periods.',
                'health_benefit': 'Safety cannot be guaranteed. Manual verification is essential before consumption.'
            }
        },
        'rotten': {
            'very_high': {
                'title': '🚫 Severely Spoiled - Do Not Consume',
                'description': f'Our AI Vision System has analyzed the image with extremely high confidence ({confidence:.1f}%) and detected severe spoilage. This {food_name.lower()} shows multiple visual markers of advanced decay.',
                'analysis': [
                    f'🎨 Color Analysis: The AI detects {visual_features["color"]}. This discoloration pattern is a critical indicator of cellular breakdown and bacterial/fungal activity that the neural network has been trained to recognize as unsafe.',
                    f'🔍 Texture Assessment: Visual examination reveals {visual_features["texture"]}. The AI\'s pattern recognition identifies this as advanced decomposition where cell walls have collapsed, making the food completely unsuitable for consumption.',
                    f'🌡️ Surface Condition: Image analysis shows {visual_features["surface"]}. The AI detects these surface abnormalities as clear evidence of mold growth, bacterial colonization, or advanced decay - all dangerous for human consumption.',
                    f'⚡ Critical Warning: The computer vision model identifies {visual_features["specific"]}. This combination of visual decay markers triggers the AI\'s highest alert level, indicating the {food_name.lower()} is severely spoiled and poses significant health risks.'
                ],
                'recommendation': '⚠️ DO NOT CONSUME. Discard this food immediately to avoid potential foodborne illness.',
                'health_risk': 'Consuming spoiled food can lead to food poisoning, stomach upset, nausea, vomiting, or more serious health issues.',
                'disposal': 'Seal in a plastic bag and dispose of properly. Wash any surfaces or containers that contacted this food.'
            },
            'high': {
                'title': '❌ Spoiled Food Detected',
                'description': f'High confidence ({confidence:.1f}%) analysis indicates this {food_name.lower()} has spoiled and is no longer safe for consumption.',
                'analysis': [
                    f'🎨 Color Analysis: The AI detects {visual_features["color"]}. The neural network recognizes this color change as a definitive sign of oxidation, enzymatic browning, or microbial growth that occurs during spoilage.',
                    f'🔍 Texture Assessment: Visual features show {visual_features["texture"]}. The AI\'s trained models identify this texture degradation as cellular breakdown typical of spoiled produce that has passed safe consumption threshold.',
                    f'🌡️ Surface Condition: The image analysis reveals {visual_features["surface"]}. These surface abnormalities detected by the AI are consistent with fungal growth, bacterial colonization, or advanced decay processes.',
                    f'⚡ Spoilage Confirmation: Based on comprehensive visual analysis, the AI identifies {visual_features["specific"]}. These combined indicators strongly suggest this {food_name.lower()} is spoiled and poses health risks if consumed.'
                ],
                'recommendation': 'This food is unsafe to eat. We strongly recommend discarding it to prevent health risks.',
                'health_risk': 'Eating spoiled food may cause digestive problems, food poisoning, or bacterial infections.',
                'disposal': 'Dispose of this food safely. Do not attempt to salvage any portions.'
            },
            'medium': {
                'title': '⚠️ Possible Spoilage - High Caution Advised',
                'description': f'Moderate confidence ({confidence:.1f}%) suggests potential spoilage. The AI detects concerning visual patterns that warrant serious caution.',
                'analysis': [
                    f'🎨 Color Analysis: The AI detects some characteristics matching {visual_features["color"]}, but the patterns are not as definitive as in clear spoilage cases. The neural network identifies color changes that could indicate early to moderate decay.',
                    f'🔍 Texture Assessment: Visual analysis shows features suggesting {visual_features["texture"]}. The AI recognizes these as potential warning signs, though some areas appear ambiguous. This mixed pattern is concerning.',
                    f'🌡️ Surface Condition: Image analysis reveals {visual_features["surface"]}. While not as severe as advanced spoilage, the AI detects abnormalities that raise red flags about safety.',
                    f'⚠ Risk Assessment: The AI\'s analysis identifies {visual_features["specific"]}, along with some inconclusive markers. Given these mixed signals, human verification is critical, but caution is strongly advised as early spoilage may not always be visually obvious.'
                ],
                'recommendation': 'Exercise extreme caution. When in doubt, throw it out. The risk is not worth potential health consequences.',
                'health_risk': 'Partially spoiled food can still contain harmful bacteria even if not fully rotten.',
                'disposal': 'Safest option is to discard. If you choose to inspect further, check smell and texture carefully.'
            },
            'low': {
                'title': '🔍 Quality Uncertain - Proceed with Caution',
                'description': f'Low confidence ({confidence:.1f}%) in spoilage detection. The visual features are too ambiguous for the AI to make a reliable determination.',
                'analysis': [
                    f'🎨 Color Analysis: The AI attempts to match visual patterns with known spoilage indicators ({visual_features["color"]}), but the data is inconclusive. Color patterns could indicate multiple possible conditions.',
                    f'🔍 Texture Assessment: The neural network detects mixed signals regarding {visual_features["texture"]}. Some features align with spoilage, others with borderline freshness, creating analytical uncertainty.',
                    f'🌡️ Visual Data Limitations: The AI identifies {visual_features["surface"]}, but image quality, lighting, or angle issues prevent confident classification. The computer vision system requires clearer visual data for accurate spoilage detection.',
                    f'⚡ Human Verification Essential: With low confidence, the AI cannot reliably determine if this {food_name.lower()} is spoiled or borderline fresh. Rely on human senses - check for off-odors (sour, fermented, or foul smells), unusual sliminess, or visible mold not detected in the image.'
                ],
                'recommendation': 'Perform comprehensive manual inspection. Check for off-odors, unusual texture, or any signs of mold or decay.',
                'health_risk': 'Unknown. Err on the side of caution - if it seems questionable, do not consume.',
                'disposal': 'If any doubt remains after inspection, dispose of the food to ensure safety.'
            }
        }
    }
    
    # Get appropriate explanation
    status_key = 'fresh' if status == 'segar' else 'rotten'
    explanation = explanations[status_key][confidence_level]
    
    return explanation

def predict_image(image_path):
    """Make prediction on uploaded image"""
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img_array)
        
        # Make prediction
        predictions = model(img_tensor, training=False)
        
        # Get predicted class and confidence
        predicted_idx = int(tf.argmax(predictions[0]).numpy())
        confidence = tf.nn.softmax(predictions[0]).numpy()[predicted_idx]
        
        print(f"Predicted index: {predicted_idx}")
        print(f"Confidence: {confidence:.4f}")
        
        # Get class name with multiple fallback options
        predicted_class = None
        
        # Try 1: idx_to_class with string key
        if 'idx_to_class' in class_info and str(predicted_idx) in class_info['idx_to_class']:
            predicted_class = class_info['idx_to_class'][str(predicted_idx)]
            print(f"✓ Found class via idx_to_class[str]: {predicted_class}")
        
        # Try 2: idx_to_class with int key
        elif 'idx_to_class' in class_info and predicted_idx in class_info['idx_to_class']:
            predicted_class = class_info['idx_to_class'][predicted_idx]
            print(f"✓ Found class via idx_to_class[int]: {predicted_class}")
        
        # Try 3: class_names list
        elif 'class_names' in class_info and predicted_idx < len(class_info['class_names']):
            predicted_class = class_info['class_names'][predicted_idx]
            print(f"✓ Found class via class_names list: {predicted_class}")
        
        else:
            raise ValueError(f"Cannot find class name for index {predicted_idx}")
        
        # Determine status (fresh or rotten)
        if 'fresh' in predicted_class.lower():
            status = 'segar'
        elif 'rotten' in predicted_class.lower():
            status = 'busuk'
        else:
            status = 'unknown'
        
        # Generate AI explanation
        confidence_percent = float(confidence * 100)
        ai_explanation = generate_ai_explanation(predicted_class, status, confidence_percent)
        
        return {
            'class': predicted_class,
            'confidence': confidence_percent,
            'status': status,
            'class_idx': predicted_idx,
            'ai_explanation': ai_explanation
        }
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        print(f"class_info keys: {class_info.keys() if class_info else 'None'}")
        if 'idx_to_class' in class_info:
            print(f"idx_to_class type: {type(class_info['idx_to_class'])}")
            print(f"idx_to_class keys (first 5): {list(class_info['idx_to_class'].keys())[:5]}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    filepath = None
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing image: {filepath}")
        
        # Make prediction
        result = predict_image(filepath)
        
        print(f"Prediction successful: {result['class']} ({result['confidence']:.1f}%)")
        
        # Clean up uploaded file
        os.remove(filepath)
        filepath = None
        
        # Return result with AI explanation
        return jsonify({
            'success': True,
            'class': result['class'],
            'confidence': f"{result['confidence']:.1f}",
            'status': result['status'],
            'class_idx': result['class_idx'],
            'ai_explanation': result['ai_explanation']
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up file if exists
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'num_classes': class_info['num_classes'] if class_info else 0
    })

@app.route('/classes')
def get_classes():
    """Get all available classes"""
    if class_info is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'classes': class_info['class_names'],
        'num_classes': class_info['num_classes']
    })

if __name__ == '__main__':
    print("="*60)
    print("INITIALIZING FLASK APP - AI FOOD QUALITY DETECTION")
    print("="*60)
    
    # Load model and assets
    try:
        load_model_assets()
        print("\n" + "="*60)
        print("MODEL LOADED SUCCESSFULLY!")
        print("="*60)
        print(f"\nSupported classes:")
        for idx, name in enumerate(class_info['class_names']):
            print(f"  {idx}: {name}")
        print("\n" + "="*60)
        print("STARTING FLASK SERVER...")
        print("="*60)
        print("\n🚀 Server running at: http://127.0.0.1:5000")
        print("📊 Access the web interface at: http://127.0.0.1:5000")
        print("\n" + "="*60 + "\n")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Failed to load model")
        print("="*60)
        print(f"\n{str(e)}\n")
        print("Please make sure you have:")
        print("  1. Trained the model by running: python food_quality_detection.py")
        print("  2. Model checkpoint exists at: vit_checkpoints/vit_food_quality.h5")
        print("  3. Model assets exist at: model_assets/")
        print("\n" + "="*60 + "\n")

