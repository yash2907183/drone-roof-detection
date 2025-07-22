import streamlit as st
from PIL import Image
import numpy as np

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

# File uploader
uploaded_file = st.file_uploader(
    "Choose an aerial image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload drone or aerial images for best results"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        if st.button('🔍 Detect Objects', type="primary"):
            with st.spinner('Analyzing image...'):
                
                if model_loaded and model is not None:
                    # REAL DETECTION with your model
                    try:
                        # Run inference
                        results = model(np.array(image), conf=0.5, verbose=False)
                        
                        # Get annotated image
                        import cv2
                        annotated_img = results[0].plot()
                        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        annotated_pil = Image.fromarray(annotated_img_rgb)
                        
                        # Display results
                        st.image(annotated_pil, caption='Detection Results', use_column_width=True)
                        
                        # Extract and display detection info
                        detections = []
                        if results[0].boxes is not None:
                            for box in results[0].boxes:
                                detection = {
                                    "class": model.names[int(box.cls)],
                                    "confidence": float(box.conf),
                                    "bbox": box.xyxy[0].tolist()
                                }
                                detections.append(detection)
                        
                        if detections:
                            st.success(f"Found {len(detections)} objects!")
                            for i, detection in enumerate(detections, 1):
                                st.write(f"**{i}.** {detection['class']} - {detection['confidence']*100:.1f}% confidence")
                        else:
                            st.warning("No objects detected. Try adjusting the image or confidence threshold.")
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")
                        st.write("Falling back to mock results...")
                        # Fall back to mock results if real detection fails
                        st.image(image, caption='Mock Results', use_column_width=True)
                
                else:
                    # MOCK RESULTS (fallback)
                    st.image(image, caption='Mock Results - Model Not Available', use_column_width=True)
                    st.write("**Mock Detected Objects:**")
                    st.write("1. **Flat Roof** - 95.0% confidence")
                    st.write("2. **Solar Panel** - 87.0% confidence")

# Instructions
st.markdown("---")
st.markdown("### How to Use:")
st.markdown("1. Upload an aerial or drone image")
st.markdown("2. Click 'Detect Objects' to run inference") 
st.markdown("3. View results with bounding boxes and confidence scores")

st.markdown("*Powered by YOLOv8 & Streamlit*")
