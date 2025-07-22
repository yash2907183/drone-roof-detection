import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# Add model loading
@st.cache_resource
def load_model():
    """Load the YOLOv8 model (cached for performance)"""
    try:
        from ultralytics import YOLO
        model = YOLO('best.pt')
        return model, True
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, False

st.set_page_config(page_title="Drone Roof Detection", page_icon="🚁")

st.title("🚁 Drone Roof & Solar Panel Detection")
st.write("Upload an aerial/drone image to detect different roof types and solar panels!")

# Load model
model, model_loaded = load_model()

# Sidebar info
st.sidebar.header("Model Status")
if model_loaded:
    st.sidebar.success("✅ Model Loaded Successfully")
    st.sidebar.write(f"**Classes:** {list(model.names.values())}")
    st.sidebar.write("**Model:** YOLOv8 Custom Trained")
    st.sidebar.write("**Accuracy:** 70.6% mAP@50")
else:
    st.sidebar.error("❌ Model Loading Failed")
    st.sidebar.write("Using mock results for demo")

# Add confidence slider
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.7,
    step=0.05,
    help="Higher = more accurate, Lower = more detections"
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an aerial image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload drone or aerial images for best results"
)

if uploaded_file is not None:
    # Load and process image
    image = Image.open(uploaded_file)
    
    # Convert RGBA to RGB if needed (fix the 4-channel issue)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        if st.button('🔍 Detect Objects', type="primary"):
            with st.spinner('Analyzing image...'):
                if model_loaded and model is not None:
                    try:
                        # Convert PIL to numpy array
                        image_np = np.array(image)
                        
                        # Run inference with confidence slider
                        results = model(image_np, conf=confidence_threshold, verbose=False)
                        
                        # Create figure with original image (preserves colors)
                        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                        ax.imshow(image)  # Use original PIL image
                        
                        # Draw bounding boxes manually
                        detections = []
                        if results[0].boxes is not None:
                            for box in results[0].boxes:
                                # Get coordinates
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                width = x2 - x1
                                height = y2 - y1
                                
                                # Draw rectangle (cyan color like YOLOv8)
                                rect = patches.Rectangle(
                                    (x1, y1), width, height,
                                    linewidth=2, edgecolor='cyan', facecolor='none'
                                )
                                ax.add_patch(rect)
                                
                                # Add label
                                class_name = model.names[int(box.cls)]
                                confidence = float(box.conf)
                                ax.text(x1, y1-5, f'{class_name} {confidence:.2f}', 
                                        color='cyan', fontsize=10, weight='bold',
                                        bbox=dict(boxstyle="round,pad=0.2", facecolor="cyan", alpha=0.7))
                                
                                detections.append({
                                    "class": class_name,
                                    "confidence": confidence,
                                })
                        
                        ax.axis('off')
                        ax.set_xlim(0, image.width)
                        ax.set_ylim(image.height, 0)
                        plt.tight_layout()
                        
                        # Convert to image and display
                        buf = BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                        buf.seek(0)
                        result_image = Image.open(buf)
                        plt.close()
                        
                        # Display with correct colors
                        st.image(result_image, caption='Detection Results', use_container_width=True)
                        
                        # Show detection info
                        if detections:
                            st.success(f"Found {len(detections)} objects!")
                            for i, detection in enumerate(detections, 1):
                                st.write(f"**{i}.** {detection['class']} - {detection['confidence']*100:.1f}% confidence")
                        else:
                            st.warning("No objects detected. Try lowering the confidence threshold.")
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")
                else:
                    st.error("Model not loaded - cannot run detection")

# Instructions
st.markdown("---")
st.markdown("### How to Use:")
st.markdown("1. Upload an aerial or drone image")
st.markdown("2. Adjust confidence threshold in sidebar (try 0.7-0.8 for best results)")
st.markdown("3. Click 'Detect Objects' to run inference") 
st.markdown("4. View results with bounding boxes and confidence scores")

st.markdown("*Powered by YOLOv8 & Streamlit*")
