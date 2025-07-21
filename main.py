import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Load model
@st.cache_resource
def load_model():
    """Load the YOLOv8 model (cached for performance)"""
    from ultralytics import YOLO
    model = YOLO('best.pt')
    return model

st.set_page_config(page_title="Drone Roof Detection", page_icon="🚁")

st.title("🚁 Drone Roof & Solar Panel Detection")
st.write("Upload an aerial/drone image to detect different roof types and solar panels!")

# Load model
model = load_model()

# Sidebar info
st.sidebar.header("Model Status")
st.sidebar.success("✅ Model Loaded Successfully")
st.sidebar.write(f"**Classes:** {list(model.names.values())}")
st.sidebar.write("**Model:** YOLOv8 Custom Trained")
st.sidebar.write("**Accuracy:** 70.6% mAP@50")

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
                # Convert PIL to numpy array
                image_np = np.array(image)
                
                # Run inference with user-defined confidence
                results = model(image_np, conf=confidence_threshold, verbose=False)
                
                # Use YOLOv8's built-in save (handles colors correctly)
                results[0].save('detection_result.jpg')

                # Load and display the saved result
                result_image = Image.open('detection_result.jpg')
                st.image(result_image, caption='Detection Results', use_container_width=True)
                
                # Extract and display detection info
                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        detection = {
                            "class": model.names[int(box.cls)],
                            "confidence": float(box.conf),
                        }
                        detections.append(detection)
                
                if detections:
                    st.success(f"Found {len(detections)} objects!")
                    for i, detection in enumerate(detections, 1):
                        st.write(f"**{i}.** {detection['class']} - {detection['confidence']*100:.1f}% confidence")
                else:
                    st.info("No objects detected. Try lowering the confidence threshold.")
                
                st.balloons()

# Instructions
st.markdown("---")
st.markdown("### How to Use:")
st.markdown("1. Upload an aerial or drone image")
st.markdown("2. Adjust confidence threshold in sidebar (try 0.7-0.8 for best results)") 
st.markdown("3. Click 'Detect Objects' to run inference")
st.markdown("4. View results with bounding boxes and confidence scores")

st.markdown("*Powered by YOLOv8 & Streamlit*")
