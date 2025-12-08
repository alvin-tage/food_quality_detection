import os, math, time, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
import six
from tensorflow.keras.callbacks import TensorBoard
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
import json
print(tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE

# Basic config for ViT
TRAIN_DIR = "./dataset_food_quality/train/"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
NUM_CLASSES = 18

# Paths for ViT training
CHECKPOINT_DIR = Path("vit_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR = Path("vit_logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = Path("vit_plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)
MODEL_ASSETS_DIR = Path("model_assets")
MODEL_ASSETS_DIR.mkdir(exist_ok=True, parents=True)

print("TF version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

#============================================================
# Dataset loader
#============================================================

def load_image(path):
    """Load and preprocess image for ViT"""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    # ImageNet normalization:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    # Transpose to (C, H, W) format for ViT
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)

# Get images and labels from subdirectories
image_paths = []
labels = []
class_names = sorted([d.name for d in Path(TRAIN_DIR).iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

print(f"Found {len(class_names)} classes: {class_names}")

for class_name in class_names:
    class_dir = Path(TRAIN_DIR) / class_name
    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            image_paths.append(str(img_path))
            labels.append(class_to_idx[class_name])

if len(image_paths) == 0:
    raise RuntimeError(f"No images found in {TRAIN_DIR}")

print(f"Found {len(image_paths)} images across {len(class_names)} classes")

def make_dataset(paths, labels, batch_size=BATCH_SIZE, shuffle=True):
    """Create dataset for ViT classification"""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))
    
    def load_and_preprocess(path, label):
        img = tf.py_function(
            lambda x: tf.convert_to_tensor(load_image(x.numpy().decode())), 
            [path], 
            Tout=tf.float32
        )
        img.set_shape([3, IMAGE_SIZE, IMAGE_SIZE])
        return img, label
    
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds

#============================================================
# Rearrange Layer
#============================================================
class Rearrange(tf.keras.layers.Layer):
    def __init__(self, pattern, **axis_sizes):
        super().__init__()
        self.pattern = pattern
        self.axis_sizes = axis_sizes
    
    def call(self, x):
        if 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)' in self.pattern:
            p1 = self.axis_sizes['p1']
            p2 = self.axis_sizes['p2']
            batch, channels, height, width = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            h = height // p1
            w = width // p2
            
            x = tf.reshape(x, [batch, channels, h, p1, w, p2])
            x = tf.transpose(x, [0, 2, 4, 3, 5, 1])
            x = tf.reshape(x, [batch, h * w, p1 * p2 * channels])
            return x
            
        elif 'b n (qkv h d) -> qkv b h n d' in self.pattern:
            qkv = self.axis_sizes['qkv']
            h = self.axis_sizes['h']
            batch, n, _ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
            d = tf.shape(x)[2] // (qkv * h)
            
            x = tf.reshape(x, [batch, n, qkv, h, d])
            x = tf.transpose(x, [2, 0, 3, 1, 4])
            return x
            
        elif 'b h n d -> b n (h d)' in self.pattern:
            batch, h, n, d = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.reshape(x, [batch, n, h * d])
            return x
        
        return x
    
#============================================================
# Activation Functions
#============================================================
def gelu(x):
    """Gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def get_activation(identifier):
    """Maps a identifier to a Python function."""
    if isinstance(identifier, six.string_types):
        name_to_fn = {"gelu": gelu}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)

#============================================================
# ViT Components
#============================================================
class Residual(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x

class PreNorm(tf.keras.Model):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fn = fn

    def call(self, x):
        return self.fn(self.norm(x))

class FeedForward(tf.keras.Model):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=get_activation('gelu')),
            tf.keras.layers.Dense(dim)
        ])

    def call(self, x):
        return self.net(x)

class Attention(tf.keras.Model):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)

        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return out

class Transformer(tf.keras.Model):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)

class ViT(tf.keras.Model):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.pos_embedding = self.add_weight(name="position_embeddings",
                                             shape=[num_patches + 1, dim],
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             dtype=tf.float32)
        self.patch_to_embedding = tf.keras.layers.Dense(dim)
        self.cls_token = self.add_weight(name="cls_token",
                                         shape=[1, 1, dim],
                                         initializer=tf.keras.initializers.RandomNormal(),
                                         dtype=tf.float32)

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = tf.identity
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=get_activation('gelu')),
            tf.keras.layers.Dense(num_classes)
        ])

    @tf.function
    def call(self, img):
        shapes = tf.shape(img)
        x = self.rearrange(img)
        x = self.patch_to_embedding(x)

        cls_tokens = tf.broadcast_to(self.cls_token, (shapes[0], 1, self.dim))
        x = tf.concat((cls_tokens, x), axis=1)
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
    
#============================================================
# Trainer Configuration and Class
#============================================================
logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    ckpt_path = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, model_config, train_dataset, train_dataset_len, test_dataset, test_dataset_len, config):
        self.train_dataset = train_dataset
        self.train_dataset_len = train_dataset_len
        self.test_dataset = test_dataset
        self.test_dataset_len = test_dataset_len
        self.config = config
        self.tokens = 0
        
        # Initialize history dictionary
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Setup distribution strategy
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        elif len(gpus) == 1:
            self.strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        else:
            self.strategy = tf.distribute.OneDeviceStrategy("CPU:0")

        with self.strategy.scope():
            self.model = model(**model_config)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            self.cce = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, 
                reduction=tf.keras.losses.Reduction.NONE
            )
            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
            if self.test_dataset:
                self.test_dist_dataset = self.strategy.experimental_distribute_dataset(self.test_dataset)

    def save_checkpoints(self):
        if self.config.ckpt_path is not None:
            self.model.save_weights(self.config.ckpt_path)

    def train(self):
        train_loss_metric = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
        test_loss_metric = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('training_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('testing_accuracy')

        @tf.function
        def train_step(dist_inputs):
            def step_fn(inputs):
                X, Y = inputs
                with tf.GradientTape() as tape:
                    logits = self.model(X, training=True)
                    loss = self.cce(Y, logits)
                    loss = tf.nn.compute_average_loss(loss, global_batch_size=self.config.batch_size)
                
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                train_accuracy.update_state(Y, logits)
                return loss

            per_replica_losses = self.strategy.run(step_fn, args=(dist_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def test_step(dist_inputs):
            def step_fn(inputs):
                X, Y = inputs
                logits = self.model(X, training=False)
                loss = self.cce(Y, logits)
                loss = tf.nn.compute_average_loss(loss, global_batch_size=self.config.batch_size)
                
                test_accuracy.update_state(Y, logits)
                return loss

            per_replica_losses = self.strategy.run(step_fn, args=(dist_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # Training loop
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            
            # Training
            for inputs in tqdm(self.train_dist_dataset, desc="Training", total=self.train_dataset_len // self.config.batch_size):
                loss = train_step(inputs)
                train_loss_metric(loss)
            
            train_loss = train_loss_metric.result()
            train_acc = train_accuracy.result()
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            
            # Store history
            self.history['train_loss'].append(float(train_loss))
            self.history['train_accuracy'].append(float(train_acc))
            
            train_loss_metric.reset_states()
            train_accuracy.reset_states()

            # Testing
            if self.test_dataset:
                for inputs in tqdm(self.test_dist_dataset, desc="Validation", total=self.test_dataset_len // self.config.batch_size):
                    loss = test_step(inputs)
                    test_loss_metric(loss)
                
                test_loss = test_loss_metric.result()
                test_acc = test_accuracy.result()
                print(f"Val Loss: {test_loss:.4f}, Val Accuracy: {test_acc:.4f}")
                
                # Store history
                self.history['val_loss'].append(float(test_loss))
                self.history['val_accuracy'].append(float(test_acc))
                
                test_loss_metric.reset_states()
                test_accuracy.reset_states()

            self.save_checkpoints()

#============================================================
# Plot Training History
#============================================================

def plot_training_history(history, save_path=None):
    """Plot training and validation loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracy'], 'r-o', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

#============================================================
# Evaluate Model
#============================================================

def evaluate_model(model, dataset, true_labels, class_names, save_dir=None):
    """Evaluate model and generate classification report and confusion matrix"""
    
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Get predictions
    all_predictions = []
    all_true_labels = []
    
    print("\nGenerating predictions...")
    for images, labels in tqdm(dataset):
        predictions = model(images, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)
        
        all_predictions.extend(predicted_labels.numpy())
        all_true_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save classification report
    if save_dir:
        report_path = Path(save_dir) / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        print(f"\nClassification report saved to: {report_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_dir:
        cm_path = Path(save_dir) / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
    
    plt.show()
    
    # Overall accuracy
    accuracy = np.mean(all_predictions == all_true_labels)
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return report, cm, accuracy

#============================================================
# Save Model Assets for Flask Deployment
#============================================================

def save_model_assets(class_names, class_to_idx, model_config, history, save_dir):
    """Save all necessary assets for Flask deployment"""
    
    print("\n" + "="*60)
    print("SAVING MODEL ASSETS FOR DEPLOYMENT")
    print("="*60)
    
    save_dir = Path(save_dir)
    
    # 1. Save class names and mapping
    class_info = {
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'num_classes': len(class_names)
    }
    
    with open(save_dir / 'class_info.pkl', 'wb') as f:
        pickle.dump(class_info, f)
    print(f"✓ Saved class_info.pkl")
    
    # Also save as JSON for easier reading
    with open(save_dir / 'class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"✓ Saved class_info.json")
    
    # 2. Save model configuration
    with open(save_dir / 'model_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)
    print(f"✓ Saved model_config.pkl")
    
    with open(save_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"✓ Saved model_config.json")
    
    # 3. Save training history
    with open(save_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f"✓ Saved training_history.pkl")
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Saved training_history.json")
    
    # 4. Save image preprocessing parameters
    preprocess_params = {
        'image_size': IMAGE_SIZE,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'input_format': 'CHW',  # Channel, Height, Width
    }
    
    with open(save_dir / 'preprocess_params.pkl', 'wb') as f:
        pickle.dump(preprocess_params, f)
    print(f"✓ Saved preprocess_params.pkl")
    
    with open(save_dir / 'preprocess_params.json', 'w') as f:
        json.dump(preprocess_params, f, indent=2)
    print(f"✓ Saved preprocess_params.json")
    
    # 5. Create README for deployment
    readme_content = f"""# ViT Food Quality Model - Deployment Assets

### Model Information:
- **Number of Classes**: {len(class_names)}
- **Input Size**: {IMAGE_SIZE}x{IMAGE_SIZE}
- **Model Architecture**: Vision Transformer (ViT)
- **Patch Size**: {model_config['patch_size']}
- **Embedding Dimension**: {model_config['dim']}
- **Depth**: {model_config['depth']}
- **Attention Heads**: {model_config['heads']}

### Classes:
{chr(10).join([f"{idx}. {name}" for idx, name in enumerate(class_names)])}

Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(save_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    print(f"✓ Saved README.md")
    
    print(f"\n{'='*60}")
    print(f"All assets saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    return True

#============================================================
# Visual Prediction Results
#============================================================

def show_predictions(model, dataset, class_names, num_samples=12):
    """Show predictions with true vs predicted labels"""
    
    for images, true_labels in dataset.take(1):
        predictions = model(images, training=False)
        predicted_labels = tf.argmax(predictions, axis=1)
        
        images = images.numpy()
        true_labels = true_labels.numpy()
        predicted_labels = predicted_labels.numpy()
        
        plt.figure(figsize=(15, 10))
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 4, i + 1)
            
            img = images[i]
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            plt.imshow(img)
            
            true_class = class_names[true_labels[i]]
            pred_class = class_names[predicted_labels[i]]
            
            color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
            plt.title(f'True: {true_class}\nPred: {pred_class}', color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        batch_accuracy = np.mean(true_labels == predicted_labels)
        print(f"Batch Accuracy: {batch_accuracy:.2%}")
        print("-" * 50)

#============================================================
# Main Training Script
#============================================================
if __name__ == "__main__":
    print("\nSplitting dataset into train/validation...")
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    print("\nCreating datasets...")
    train_dataset = make_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = make_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)

    model_config = {
        'image_size': IMAGE_SIZE,
        'patch_size': 16,
        'num_classes': len(class_names),
        'dim': 256,
        'depth': 8,
        'heads': 8,
        'mlp_dim': 512,
        'channels': 3
    }

    tconf = TrainerConfig(
        max_epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        learning_rate=LEARNING_RATE,
        ckpt_path=str(CHECKPOINT_DIR / 'vit_food_quality.h5')
    )

    print("\nModel Configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {tconf.max_epochs}")
    print(f"  Batch Size: {tconf.batch_size}")
    print(f"  Learning Rate: {tconf.learning_rate}")
    
    print(f"\nClass mapping:")
    for name, idx in class_to_idx.items():
        print(f"  {idx}: {name}")

    print("\nInitializing trainer...")
    trainer = Trainer(
        ViT, 
        model_config, 
        train_dataset, 
        len(train_paths), 
        val_dataset, 
        len(val_paths), 
        tconf
    )

    print("\nStarting training...")
    trainer.train()
    print("\nTraining completed!")
    print(f"Model saved to: {tconf.ckpt_path}")
    
    #============================================================
    # Plot Training History
    #============================================================
    print("\n" + "="*60)
    print("PLOTTING TRAINING HISTORY")
    print("="*60)
    plot_training_history(
        trainer.history, 
        save_path=PLOTS_DIR / 'training_history.png'
    )
    
    #============================================================
    # Evaluate Model
    #============================================================
    report, cm, accuracy = evaluate_model(
        trainer.model, 
        val_dataset, 
        val_labels, 
        class_names,
        save_dir=PLOTS_DIR
    )
    
    #============================================================
    # Save Model Assets for Flask Deployment
    #============================================================
    save_model_assets(
        class_names=class_names,
        class_to_idx=class_to_idx,
        model_config=model_config,
        history=trainer.history,
        save_dir=MODEL_ASSETS_DIR
    )
    
    #============================================================
    # Visual Prediction Results
    #============================================================
    print("\n" + "="*60)
    print("VISUALIZING PREDICTIONS")
    print("="*60)
    show_predictions(trainer.model, val_dataset, class_names)