import streamlit as st
import cv2
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from isro_sr.models import DualEncoderGANSR
from isro_sr.utils.image_utils import align_images, compute_quality_metrics

# Set page config
st.set_page_config(
    page_title="Satellite Image Super-Resolution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0e566e;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e81b0;
        margin-bottom: 0.5rem;
    }
    .comparison-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .image-caption {
        text-align: center;
        font-weight: 500;
        color: #333;
    }
    .metrics-container {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the pre-trained SR model"""
    # In a real implementation, you would use an actual path to your model
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dual_encoder_gan_sr.pt')
        if os.path.exists(model_path):
            model = DualEncoderGANSR.load_from_checkpoint(model_path)
        else:
            # Placeholder for model not found
            st.warning("Model file not found. Using a mock model for demonstration.")
            # Create a dummy model for demonstration
            model = type('DummyModel', (), {
                'eval': lambda: None,
                'to': lambda device: model,
                'generate_sr': lambda img1, img2, augment: np.maximum(img1, img2)  # Simple mock function
            })
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_images(img1, img2, target_size=(256, 256)):
    """Preprocess and resize images"""
    img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_CUBIC)
    return img1, img2

def run_super_resolution(model, img1, img2, use_augmentation=False):
    """Run super-resolution on the image pair"""
    if model is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
    with torch.no_grad():
        model.eval()
        result = model.generate_sr(img1, img2, augment=use_augmentation)
        
    # Convert to uint8 for display
    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy().transpose(1, 2, 0)
        result = (result * 255).astype(np.uint8)
    
    return result

def plot_quality_metrics(metrics):
    """Create a bar chart for the quality metrics"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    metrics_name = list(metrics.keys())
    metrics_value = list(metrics.values())
    
    bars = ax.bar(metrics_name, metrics_value, color=['#1e81b0', '#3498db'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylabel('Score (Lower is better)')
    ax.set_title('Image Quality Assessment')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

def generate_report(img1, img2, sr_img, metrics):
    """Generate a downloadable report"""
    report = io.StringIO()
    report.write("# Super-Resolution Report\n\n")
    report.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    report.write("## Quality Metrics\n\n")
    for metric, value in metrics.items():
        report.write(f"- {metric}: {value:.4f}\n")
    
    report.write("\n## Processing Information\n\n")
    report.write(f"- Input Image 1 Size: {img1.shape[1]}x{img1.shape[0]}\n")
    report.write(f"- Input Image 2 Size: {img2.shape[1]}x{img2.shape[0]}\n")
    report.write(f"- Output SR Image Size: {sr_img.shape[1]}x{sr_img.shape[0]}\n")
    
    return report.getvalue()

def main():
    # Title Section
    st.markdown("<h1 class='main-header'>Dual Image Super-Resolution for Satellite Imagery</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        This application uses a hybrid classical + GAN pipeline to perform super-resolution 
        on satellite imagery using two low-resolution input images. The system aligns the images, 
        applies a dual-encoder GAN model, and provides blind quality evaluation metrics.
        """
    )
    
    # Optionally display a flowchart if available
    flowchart_path = os.path.join(os.path.dirname(__file__), 'flowchart.png')
    if os.path.exists(flowchart_path):
        with st.expander("View Project Flowchart", expanded=False):
            st.image(flowchart_path, caption="Project Pipeline Flowchart")
    
    # Create tabs for the main sections
    tabs = st.tabs(["Upload & Process", "Results & Evaluation", "Model Details"])
    
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Upload Images</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file1 = st.file_uploader("Upload first low-resolution image", type=["jpg", "jpeg", "png"])
            if uploaded_file1 is not None:
                img1 = Image.open(uploaded_file1)
                img1_np = np.array(img1)
                st.image(img1, caption="Input Image 1", use_column_width=True)
            else:
                img1_np = None
        
        with col2:
            uploaded_file2 = st.file_uploader("Upload second low-resolution image", type=["jpg", "jpeg", "png"])
            if uploaded_file2 is not None:
                img2 = Image.open(uploaded_file2)
                img2_np = np.array(img2)
                st.image(img2, caption="Input Image 2", use_column_width=True)
            else:
                img2_np = None
        
        # Preprocessing and Registration
        if img1_np is not None and img2_np is not None:
            st.markdown("<h2 class='sub-header'>Preprocessing & Registration</h2>", unsafe_allow_html=True)
            
            with st.spinner("Aligning and preprocessing images..."):
                # Preprocess and resize
                img1_processed, img2_processed = preprocess_images(img1_np, img2_np)
                
                # Image alignment
                try:
                    aligned_img1, aligned_img2 = align_images(img1_processed, img2_processed)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(aligned_img1, caption="Aligned Image 1", use_column_width=True)
                    with col2:
                        st.image(aligned_img2, caption="Aligned Image 2", use_column_width=True)
                    
                    st.success("Images aligned successfully!")
                except Exception as e:
                    st.error(f"Error during image alignment: {e}")
                    aligned_img1, aligned_img2 = img1_processed, img2_processed
            
            # Super-Resolution options
            st.markdown("<h2 class='sub-header'>Super-Resolution Options</h2>", unsafe_allow_html=True)
            
            use_augmentation = st.checkbox("Enable Mixed-Pixel Augmentation", value=False,
                                         help="Apply mixed-pixel augmentation to potentially improve results")
            
            if st.button("Generate Super-Resolution Image"):
                with st.spinner("Loading model and generating super-resolution image..."):
                    # Load model
                    model = load_model()
                    
                    if model is not None:
                        # Run super-resolution
                        sr_image = run_super_resolution(model, aligned_img1, aligned_img2, use_augmentation)
                        
                        # Store results in session state for the Results tab
                        st.session_state.sr_image = sr_image
                        st.session_state.img1 = aligned_img1
                        st.session_state.img2 = aligned_img2
                        
                        # Compute quality metrics
                        metrics = compute_quality_metrics(sr_image)
                        st.session_state.metrics = metrics
                        
                        st.success("Super-resolution image generated successfully! Check the Results tab.")
                    else:
                        st.error("Failed to load model. Please try again.")
    
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)
        
        if 'sr_image' in st.session_state:
            # Display the super-resolution result
            st.image(st.session_state.sr_image, caption="Super-Resolution Result", use_column_width=True)
            
            # Show comparison
            st.markdown("<h3>Side-by-Side Comparison</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(st.session_state.img1, caption="Input Image 1")
            
            with col2:
                st.image(st.session_state.img2, caption="Input Image 2")
            
            with col3:
                st.image(st.session_state.sr_image, caption="SR Output")
            
            # Quality Metrics
            st.markdown("<h2 class='sub-header'>Quality Evaluation</h2>", unsafe_allow_html=True)
            
            if 'metrics' in st.session_state:
                st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                
                # Display metrics as a chart
                metrics_chart = plot_quality_metrics(st.session_state.metrics)
                st.pyplot(metrics_chart)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Metrics explanation
                with st.expander("About these metrics", expanded=False):
                    st.markdown("""
                    - **NIQE (Natural Image Quality Evaluator)**: A no-reference image quality score. Lower values indicate better perceptual quality.
                    - **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)**: Measures the naturalness of an image based on measured statistical features. Lower values indicate better quality.
                    """)
            
            # Download section
            st.markdown("<h2 class='sub-header'>Download Results</h2>", unsafe_allow_html=True)
            
            # Convert SR image to bytes for download
            sr_image_pil = Image.fromarray(st.session_state.sr_image)
            buf = io.BytesIO()
            sr_image_pil.save(buf, format="PNG")
            sr_image_bytes = buf.getvalue()
            
            # Generate report
            if 'metrics' in st.session_state:
                report_text = generate_report(
                    st.session_state.img1, 
                    st.session_state.img2, 
                    st.session_state.sr_image, 
                    st.session_state.metrics
                )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download SR Image",
                    data=sr_image_bytes,
                    file_name="super_resolution_result.png",
                    mime="image/png"
                )
            
            with col2:
                if 'metrics' in st.session_state:
                    st.download_button(
                        label="Download Quality Report",
                        data=report_text,
                        file_name="quality_report.md",
                        mime="text/plain"
                    )
        else:
            st.info("Please upload images and generate a super-resolution result in the 'Upload & Process' tab.")
    
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Model Architecture</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        The Dual-Encoder GAN Super-Resolution model combines traditional computer vision techniques 
        with deep learning to achieve high-quality super-resolution from multiple low-resolution inputs.
        
        **Key Components:**
        - **Dual Encoder**: Processes two aligned low-resolution images to extract complementary features
        - **Attention Fusion**: Combines features from both images with attention mechanisms
        - **GAN Architecture**: Uses a generator-discriminator setup for realistic outputs
        - **Perceptual Loss**: Incorporates VGG-based perceptual loss for better visual quality
        """)
        
        # Expandable sections for technical details
        with st.expander("Technical Architecture Details", expanded=False):
            st.markdown("""
            - **Generator**: Dual-encoder U-Net with residual blocks and attention mechanisms
            - **Discriminator**: PatchGAN discriminator for evaluating local and global image quality
            - **Training Procedure**: Adversarial training with perceptual and L1 loss components
            - **Upscaling Factor**: 2x or 4x depending on the trained model version
            """)
        
        with st.expander("Process Flow Summary", expanded=False):
            st.markdown("""
            1. **Image Registration**: Aligns input images using ECC and Phase Correlation
            2. **Preprocessing**: Normalizes and prepares images for the model
            3. **Feature Extraction**: Each encoder extracts features from its respective input
            4. **Feature Fusion**: Attention mechanism combines complementary information
            5. **Super-Resolution**: Generator produces the high-resolution output
            6. **Post-processing**: Final adjustments for display and analysis
            """)
    
    # Footer with credits
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.markdown("Developed by the ISRO Satellite Image Super-Resolution Team", unsafe_allow_html=True)
    st.markdown("© 2023 ISRO - All Rights Reserved", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 