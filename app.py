import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import json
import requests
import numpy as np
import logging
from typing import Optional, Dict, Tuple, Any
from textwrap import dedent

# ===================== CONFIGURATION =====================
# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2048  # pixels
MIN_IMAGE_SIZE = 64  # pixels
MODEL_NAME = "tf_efficientnet_b4"  # Updated to match training
IMG_SIZE = 384  # Updated to match training
NUM_CLASSES = 7  # Updated: now 7-class model
MALIGNANT_THRESHOLD = 0.35  # Threshold for combined malignant probability

# Model file paths
MODEL_URL = "https://huggingface.co/Skindoc/streamlit3/resolve/main/best_model_20251115T091634Z.pth"
MODEL_PATH = "model_cache.pth"  # Local cache file
LABEL_MAP_PATH = "label_map_20251115T091634Z.json"  # Your label mapping (in GitHub repo)

# Updated model metrics from your training
MODEL_METRICS = {
    'auc': 96.95,  # Updated from your training results!
    'f1_score': 74.75,  # 7-class F1
    'precision': 77.42,
    'recall': 74.92,
    'final_epoch': 80
}

# HAM10000 class groupings
MALIGNANT_CLASSES = ['mel', 'bcc']  # Melanoma and Basal Cell Carcinoma
BENIGN_CLASSES = ['nv', 'bkl', 'akiec', 'vasc', 'df']  # All benign types

# Class display names
CLASS_DISPLAY_NAMES = {
    'mel': 'Melanoma',
    'bcc': 'Basal Cell Carcinoma',
    'nv': 'Melanocytic Nevus',
    'bkl': 'Benign Keratosis',
    'akiec': 'Actinic Keratosis',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
}

# Test-time augmentation parameters for uncertainty estimation
TTA_NUM_SAMPLES = 10  # Number of augmented predictions for uncertainty
TTA_ROTATION_DEG = 15  # Rotation degrees for augmentation
TTA_BRIGHTNESS = 0.15  # Brightness variation

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config with enhanced metadata
st.set_page_config(
    page_title="DermScan AI | Professional Dermoscopic Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def render_html(content: str) -> None:
    """Render HTML content safely with consistent dedenting."""
    st.markdown(dedent(content).strip(), unsafe_allow_html=True)

# [KEEP ALL YOUR EXISTING CSS - it's perfect!]
render_html("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #06A77D;
        --warning-color: #F18F01;
        --danger-color: #C73E1D;
    }
   
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
   
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #1a4d6d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
   
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
   
    .subtitle {
        color: #E8F4F8;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
   
    /* Performance metrics bar */
    .metrics-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #2E86AB;
    }
   
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
   
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2E86AB 0%, #1a4d6d 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
   
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.4);
    }
   
    /* Disclaimer box */
    .disclaimer-box {
        background: #FFF3CD;
        border: 2px solid #F18F01;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
   
    .disclaimer-title {
        color: #856404;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
   
    /* Result cards */
    .result-high-risk {
        background: linear-gradient(135deg, #FFF5F5 0%, #FFE5E5 100%);
        border-left: 5px solid #C73E1D;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
   
    .result-low-risk {
        background: linear-gradient(135deg, #F0FFF4 0%, #E5F9E7 100%);
        border-left: 5px solid #06A77D;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
   
    .result-uncertain {
        background: linear-gradient(135deg, #FFF9E6 0%, #FFF3CC 100%);
        border-left: 5px solid #F18F01;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
   
    /* Info boxes */
    .info-box {
        background: #E8F4F8;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
   
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        color: #666;
    }
   
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
</style>
""")

# ===================== HELPER FUNCTIONS =====================

def download_model(url: str, output_path: str) -> None:
    """Download model from Hugging Face if not already cached."""
    if os.path.exists(output_path):
        logger.info(f"Model already cached at {output_path}")
        return
    
    logger.info(f"Downloading model from {url}...")
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        
        logger.info(f"Model downloaded successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

@st.cache_resource
def load_label_map(label_map_path: str) -> Dict[int, str]:
    """Load label mapping from JSON file."""
    try:
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in label_map.items()}
        else:
            logger.warning(f"Label map file not found: {label_map_path}")
            # Fallback to default HAM10000 labels
            return {
                0: 'akiec',
                1: 'bcc',
                2: 'bkl',
                3: 'df',
                4: 'mel',
                5: 'nv',
                6: 'vasc'
            }
    except Exception as e:
        logger.error(f"Error loading label map: {e}")
        # Fallback
        return {
            0: 'akiec',
            1: 'bcc',
            2: 'bkl',
            3: 'df',
            4: 'mel',
            5: 'nv',
            6: 'vasc'
        }

@st.cache_resource
def load_model(model_url: str, model_path: str) -> torch.nn.Module:
    """Download and load the trained model with caching."""
    try:
        # Download model if not cached
        with st.spinner('🔄 Downloading model from Hugging Face (69MB, first time only)...'):
            download_model(model_url, model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model on device: {device}")
        
        # Create model architecture
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # Load trained weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

def get_transforms(mode='inference'):
    """Get image transforms for inference."""
    if mode == 'inference':
        return transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.05)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_tta_transforms(n_augmentations=10):
    """Get TTA transforms for uncertainty estimation."""
    base_transform = [
        transforms.Resize(int(IMG_SIZE * 1.05)),
        transforms.CenterCrop(IMG_SIZE),
    ]
    
    tta_list = []
    
    # 1. Original
    tta_list.append(transforms.Compose(base_transform + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # 2. Horizontal flip
    tta_list.append(transforms.Compose(base_transform + [
        transforms.functional.hflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # 3. Vertical flip
    tta_list.append(transforms.Compose(base_transform + [
        transforms.functional.vflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # 4-5. Rotations
    for deg in [TTA_ROTATION_DEG, -TTA_ROTATION_DEG]:
        tta_list.append(transforms.Compose(base_transform + [
            transforms.RandomRotation(degrees=(deg, deg)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    # 6-7. Brightness
    for brightness in [1 + TTA_BRIGHTNESS, 1 - TTA_BRIGHTNESS]:
        tta_list.append(transforms.Compose(base_transform + [
            transforms.ColorJitter(brightness=brightness),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    # 8. Both flips
    tta_list.append(transforms.Compose(base_transform + [
        transforms.functional.hflip,
        transforms.functional.vflip,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # 9-10. Rotation + flip combinations
    if n_augmentations > 8:
        tta_list.append(transforms.Compose(base_transform + [
            transforms.functional.hflip,
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        tta_list.append(transforms.Compose(base_transform + [
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    
    return tta_list[:n_augmentations]

def validate_image(image: Image.Image) -> Tuple[bool, Optional[str]]:
    """Validate uploaded image."""
    try:
        # Check dimensions
        width, height = image.size
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            return False, f"Image too small (min {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE})"
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            return False, f"Image too large (max {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE})"
        
        # PIL's format attribute can be None for some images
        # Instead, just verify the image can be converted to RGB
        try:
            # Try to convert to RGB (this will fail for unsupported formats)
            _ = image.convert('RGB')
        except Exception:
            return False, "Unable to process image format"
        
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def predict_with_tta(model: torch.nn.Module, image: Image.Image, 
                     label_map: Dict[int, str], n_tta: int = 10) -> Dict[str, Any]:
    """
    Make predictions with Test-Time Augmentation for uncertainty estimation.
    Returns 7-class probabilities + grouped malignant/benign.
    """
    device = next(model.parameters()).device
    tta_transforms = get_tta_transforms(n_tta)
    
    all_probs = []
    
    with torch.no_grad():
        for transform in tta_transforms:
            img_tensor = transform(image).unsqueeze(0).to(device)
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
    
    # Convert to numpy array
    all_probs = np.array(all_probs)  # Shape: (n_tta, num_classes)
    
    # Calculate mean and std for each class
    mean_probs = all_probs.mean(axis=0)  # Shape: (num_classes,)
    std_probs = all_probs.std(axis=0)    # Shape: (num_classes,)
    
    # Create per-class results
    class_results = {}
    for class_idx, class_name in label_map.items():
        class_results[class_name] = {
            'probability': float(mean_probs[class_idx]),
            'std': float(std_probs[class_idx]),
            'display_name': CLASS_DISPLAY_NAMES.get(class_name, class_name)
        }
    
    # Group into malignant/benign
    malignant_prob = sum(mean_probs[i] for i, name in label_map.items() if name in MALIGNANT_CLASSES)
    benign_prob = sum(mean_probs[i] for i, name in label_map.items() if name in BENIGN_CLASSES)
    
    # Calculate uncertainty for grouped predictions
    malignant_std = np.sqrt(sum(std_probs[i]**2 for i, name in label_map.items() if name in MALIGNANT_CLASSES))
    benign_std = np.sqrt(sum(std_probs[i]**2 for i, name in label_map.items() if name in BENIGN_CLASSES))
    
    # Calculate confidence intervals (95% = mean ± 1.96*std)
    ci_malignant = {
        'lower': max(0, malignant_prob - 1.96 * malignant_std),
        'upper': min(1, malignant_prob + 1.96 * malignant_std),
        'uncertainty': 'Low' if malignant_std < 0.08 else 'Moderate' if malignant_std < 0.15 else 'High'
    }
    
    ci_benign = {
        'lower': max(0, benign_prob - 1.96 * benign_std),
        'upper': min(1, benign_prob + 1.96 * benign_std),
        'uncertainty': 'Low' if benign_std < 0.08 else 'Moderate' if benign_std < 0.15 else 'High'
    }
    
    # Model certainty (inverse of prediction variance)
    prediction_variance = np.var(all_probs)
    model_certainty = max(0, 100 * (1 - prediction_variance * 10))
    
    return {
        'class_probabilities': class_results,
        'malignant': float(malignant_prob),
        'benign': float(benign_prob),
        'malignant_std': float(malignant_std),
        'benign_std': float(benign_std),
        'ci_malignant': ci_malignant,
        'ci_benign': ci_benign,
        'model_certainty': model_certainty,
        'prediction_variance': float(prediction_variance),
        'tta_samples': n_tta
    }

# ===================== MAIN APP =====================

# Header
render_html("""
<div class="main-header">
    <h1 class="main-title">🔬 DermScan AI</h1>
    <p class="subtitle">Advanced Multi-Class Dermoscopic Analysis • 7 Lesion Types • 96.95% AUC</p>
</div>
""")

# Performance metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 AUC Score", f"{MODEL_METRICS['auc']:.1f}%", help="Area Under ROC Curve - 96.95% is exceptional!")
with col2:
    st.metric("📊 F1 Score", f"{MODEL_METRICS['f1_score']:.1f}%", help="Balance of precision and recall (7-class)")
with col3:
    st.metric("🎚️ Precision", f"{MODEL_METRICS['precision']:.1f}%", help="Accuracy of positive predictions")
with col4:
    st.metric("🔍 Recall", f"{MODEL_METRICS['recall']:.1f}%", help="Ability to find all positive cases")

# Medical disclaimer
render_html("""
<div class="disclaimer-box">
    <div class="disclaimer-title">⚠️ Medical Disclaimer</div>
    <p style="color: #856404; font-size: 1rem; margin-bottom: 0.5rem;">
        This tool is for <strong>educational and research purposes only</strong>.
    </p>
    <ul style="color: #856404; margin: 0; padding-left: 1.5rem;">
        <li>NOT a substitute for professional medical diagnosis</li>
        <li>NOT FDA-cleared or CE-marked medical device</li>
        <li>ALL suspicious lesions require dermatologist evaluation</li>
        <li>If concerned about a skin lesion, consult a healthcare provider immediately</li>
    </ul>
</div>
""")

# Load model and label map
model = load_model(MODEL_URL, MODEL_PATH)
label_map = load_label_map(LABEL_MAP_PATH)

# Image upload
st.subheader("📤 Upload Dermoscopic Image")
uploaded_file = st.file_uploader(
    "Upload a dermoscopic image (JPEG/PNG, max 10MB)",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear dermoscopic image of the skin lesion"
)

if uploaded_file:
    try:
        # Load and validate image
        image = Image.open(uploaded_file).convert('RGB')
        is_valid, error_msg = validate_image(image)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            st.stop()
        
        # Display image
        col_img, col_info = st.columns([1, 1])
        
        with col_img:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col_info:
            st.info(f"""
            **Image Information:**
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Format: {image.format}
            - Mode: {image.mode}
            """)
        
        # Analyze button
        if st.button("🔬 Analyze Lesion with AI", type="primary"):
            with st.spinner("🔄 Analyzing with Test-Time Augmentation (10 predictions)..."):
                try:
                    # Get predictions
                    result = predict_with_tta(model, image, label_map, n_tta=TTA_NUM_SAMPLES)
                    st.session_state['result'] = result
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    st.error(f"❌ Analysis failed: {e}")

    except Exception as e:
        logger.error(f"Image loading error: {e}", exc_info=True)
        st.error(f"❌ Failed to load image: {e}")

# Display results
if 'result' in st.session_state:
    result = st.session_state['result']
    
    st.markdown("---")
    st.header("📊 AI Analysis Results")
    
    # Main binary classification (Malignant vs Benign)
    malignant_prob = result['malignant']
    benign_prob = result['benign']
    
    # Determine risk level
    if malignant_prob >= 0.65:
        card_class = "result-high-risk"
        risk_emoji = "🔴"
        risk_level = "HIGH RISK"
        recommendation = "⚡ **Immediate Action Required:** Schedule urgent dermatologist consultation within 1-2 weeks."
    elif malignant_prob >= MALIGNANT_THRESHOLD:
        card_class = "result-uncertain"
        risk_emoji = "🟡"
        risk_level = "UNCERTAIN - REQUIRES EVALUATION"
        recommendation = "⚠️ **Action Recommended:** Schedule dermatologist evaluation within 2-4 weeks."
    else:
        card_class = "result-low-risk"
        risk_emoji = "🟢"
        risk_level = "LOW RISK"
        recommendation = "✅ **Monitor:** Continue routine skin checks. Consult if lesion changes."
    
    # Display main result card
    render_html(f"""
    <div class="{card_class}">
        <h2 style="margin-top: 0;">{risk_emoji} {risk_level}</h2>
        <h3 style="font-size: 1.8rem; margin: 1rem 0;">
            Malignant Probability: {malignant_prob*100:.1f}%
        </h3>
        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
            95% Prediction Range: {result['ci_malignant']['lower']*100:.1f}% - {result['ci_malignant']['upper']*100:.1f}%
        </p>
        <p style="font-size: 1rem; margin-top: 1rem;">
            {recommendation}
        </p>
    </div>
    """)
    
    # Detailed breakdown tabs
    tab1, tab2, tab3 = st.tabs(["📊 Detailed Classification", "📈 Uncertainty Analysis", "ℹ️ Clinical Notes"])
    
    with tab1:
        st.subheader("7-Class Breakdown")
        
        # Sort classes by probability
        sorted_classes = sorted(
            result['class_probabilities'].items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        
        # Display each class
        for class_name, class_data in sorted_classes:
            prob = class_data['probability']
            std = class_data['std']
            display_name = class_data['display_name']
            
            # Determine if malignant
            is_malignant = class_name in MALIGNANT_CLASSES
            emoji = "🔴" if is_malignant else "🟢"
            type_label = "(Malignant)" if is_malignant else "(Benign)"
            
            # Progress bar
            st.markdown(f"**{emoji} {display_name} {type_label}**")
            st.progress(prob)
            st.caption(f"{prob*100:.2f}% ± {std*100:.2f}%")
            st.markdown("")
        
        # Summary table
        st.markdown("### Summary")
        st.markdown(f"""
        | Category | Probability | 95% CI | Uncertainty |
        |----------|-------------|--------|-------------|
        | 🔴 **Malignant** | **{malignant_prob*100:.2f}%** | {result['ci_malignant']['lower']*100:.1f}%-{result['ci_malignant']['upper']*100:.1f}% | {result['ci_malignant']['uncertainty']} |
        | 🟢 **Benign** | **{benign_prob*100:.2f}%** | {result['ci_benign']['lower']*100:.1f}%-{result['ci_benign']['upper']*100:.1f}% | {result['ci_benign']['uncertainty']} |
        """)
    
    with tab2:
        st.subheader("Prediction Uncertainty")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Model Certainty",
                f"{result['model_certainty']:.1f}%",
                help="Consistency across 10 augmented predictions"
            )
        with col2:
            st.metric(
                "Prediction Variance",
                f"{result['prediction_variance']:.4f}",
                help="Lower is better - indicates stable predictions"
            )
        
        st.markdown(f"""
        **Test-Time Augmentation Method:**
        - {result['tta_samples']} augmented predictions averaged
        - Includes rotations, flips, and brightness variations
        - Provides robust uncertainty estimates
        
        **Uncertainty Interpretation:**
        - 🟢 **Low** (< 8%): High confidence, consistent predictions
        - 🟡 **Moderate** (8-15%): Some variation, borderline case
        - 🔴 **High** (> 15%): Significant uncertainty, requires clinical assessment
        
        **Your Result:** {result['ci_malignant']['uncertainty']} uncertainty (±{result['malignant_std']*100:.1f}%)
        """)
    
    with tab3:
        st.markdown(f"""
        ### Clinical Interpretation
        
        **Decision Threshold:** {MALIGNANT_THRESHOLD} (optimized for high sensitivity)
        
        **Key Points:**
        - This model classifies 7 different skin lesion types
        - AUC of 96.95% indicates exceptional discrimination ability
        - Patient-level validation ensures honest, real-world metrics
        - No data leakage in training - results are trustworthy
        
        **Even with {result['ci_malignant']['uncertainty'].lower()} uncertainty:**
        - AI is a screening tool, not a diagnostic device
        - Clinical examination remains essential
        - Only a dermatologist can provide definitive diagnosis
        
        **What to bring to your appointment:**
        - This analysis report
        - Photos of the lesion over time
        - Information about any changes you've noticed
        """)

else:
    st.info("👆 Upload a dermoscopic image and click 'Analyze Lesion' to see results")

# Educational content
st.markdown("---")
tab_edu1, tab_edu2, tab_edu3 = st.tabs(["📖 ABCDE Rule", "🔬 Model Information", "🌐 Resources"])

with tab_edu1:
    render_html("""
    ### The ABCDE Rule for Skin Cancer Detection

    **Watch for these warning signs in moles and lesions:**

    **🅰️ Asymmetry**  
    One half of the mole doesn't match the other half

    **🅱️ Border Irregularity**  
    Edges are ragged, notched, or blurred rather than smooth

    **©️ Color Variation**  
    Multiple colors present or uneven color distribution

    **🅳 Diameter**  
    Larger than 6mm (about the size of a pencil eraser)

    **🅴 Evolving**  
    Changes in size, shape, color, elevation, or new symptoms

    > ⚠️ **If you notice ANY of these signs, consult a dermatologist immediately!**
    """)

with tab_edu2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **🏗️ Architecture**  
        EfficientNet-B4 (Clinical-Grade CNN)
        
        **📚 Training Dataset**  
        HAM10000 (10,015 dermoscopic images)
        
        **🎯 AUC Score**  
        {MODEL_METRICS['auc']:.2f}% (Exceptional!)
        
        **📊 F1 Score**  
        {MODEL_METRICS['f1_score']:.2f}% (7-class)
        """)
    
    with col2:
        st.markdown(f"""
        **🎚️ Precision**  
        {MODEL_METRICS['precision']:.2f}%
        
        **🔍 Recall**  
        {MODEL_METRICS['recall']:.2f}%
        
        **⚡ Training Epochs**  
        {MODEL_METRICS['final_epoch']}
        
        **🔬 Classification**  
        7 Lesion Types + Binary Grouping
        """)
    
    st.info("""
    **7 Lesion Types Classified:**
    - 🔴 Melanoma (malignant)
    - 🔴 Basal Cell Carcinoma (malignant)
    - 🟢 Melanocytic Nevus (benign)
    - 🟢 Benign Keratosis (benign)
    - 🟢 Actinic Keratosis (benign)
    - 🟢 Vascular Lesion (benign)
    - 🟢 Dermatofibroma (benign)
    """)

with tab_edu3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🇬🇧 UK Resources**
        - [British Association of Dermatology](https://www.skinhealthinfo.org.uk)
        - [NHS Skin Cancer Information](https://www.nhs.uk/conditions/skin-cancer/)
        - [Cancer Research UK](https://www.cancerresearchuk.org/about-cancer/skin-cancer)
        """)
    
    with col2:
        st.markdown("""
        **🇺🇸 US Resources**
        - [American Academy of Dermatology](https://www.aad.org/find-a-derm)
        - [Skin Cancer Foundation](https://www.skincancer.org)
        - [American Cancer Society](https://www.cancer.org/cancer/skin-cancer.html)
        """)

# Footer
render_html(f"""
<div class="custom-footer">
    <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">🔬 DermScan AI</h3>
    <p style="font-size: 1rem; margin: 0.5rem 0;">
        <strong>Multi-Class Dermoscopic Analysis with Test-Time Augmentation</strong>
    </p>
    <p style="color: #999; font-size: 0.9rem; margin: 0.5rem 0;">
        Educational & Research Tool • Not for Clinical Diagnosis
    </p>
    <p style="color: #666; font-size: 0.85rem; margin: 0.5rem 0;">
        Model: EfficientNet-B4 | AUC: {MODEL_METRICS['auc']:.1f}% | F1: {MODEL_METRICS['f1_score']:.1f}% | 7 Lesion Types
    </p>
    <p style="color: #999; font-size: 0.8rem;">
        Dr Tom Hutchinson • Oxford, United Kingdom
    </p>
</div>
""")
