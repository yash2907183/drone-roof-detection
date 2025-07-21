import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Drone Roof Detection", page_icon="🚁")

st.title("🚁 Drone Roof & Solar Panel Detection")
st.write("Upload an aerial/drone image to detect different roof types and solar panels!")

# Add some info about the model
st.sidebar.header("Model Info")
st.sidebar.write("**Model:** YOLOv8 Custom Trained")
st.sidebar.write("**Classes:** Roof Types & Solar Panels")
st.sidebar.write("**Accuracy:** 70.6% mAP@50")

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
                # Simulate processing time
                import time
                time.sleep(2)
                
                # Show success message
                st.success("Detection complete!")
                
                # Mock detection results (we'll add real model later)
                st.write("**Detected Objects:**")
                
                # Create mock bounding boxes overlay
                st.image(image, caption='Results (Mock - Model Loading Soon)', use_column_width=True)
                
                # Mock results
                detections = [
                    {"class": "Flat Roof", "confidence": 0.95, "bbox": "123,45,678,234"},
                    {"class": "Solar Panel", "confidence": 0.87, "bbox": "234,67,456,189"},
                    {"class": "Gabled Roof", "confidence": 0.82, "bbox": "345,123,567,345"}
                ]
                
                for i, detection in enumerate(detections, 1):
                    st.write(f"**{i}.** {detection['class']} - {detection['confidence']*100:.1f}% confidence")
                
                st.balloons()
                
                # Add download button for results
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=str(detections),
                    file_name="detection_results.json",
                    mime="application/json"
                )

# Add instructions
st.markdown("---")
st.markdown("### How to Use:")
st.markdown("1. Upload an aerial or drone image using the file uploader above")
st.markdown("2. Click the 'Detect Objects' button")
st.markdown("3. View the detection results with confidence scores")
st.markdown("4. Download the results if needed")

st.markdown("### Coming Soon:")
st.markdown("- Real-time model inference")
st.markdown("- Batch processing")
st.markdown("- API endpoints")

# Footer
st.markdown("---")
st.markdown("*Powered by YOLOv8 & Streamlit*")
