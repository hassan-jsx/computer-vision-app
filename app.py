import streamlit as st
import time
from image_processing import gaussian_filter, butterworth_filter, laplacian_filter, histogram_match
import numpy as np
from PIL import Image

st.header('Computer vision app', divider='gray')

# File uploader for the first image
uploaded_file1 = st.file_uploader("Choose the first PNG or JPEG file", type=["png", "jpeg"])

# Algorithm selection
option = st.selectbox(
    "Select algorithm",
    ("Lowpass Gaussian filter (spatial domain)", "Lowpass Butterworth filter (Frequency domain)", "HighPass Laplacian filter (spatial Domain)", "Histogram Matching")
)

# Additional file uploader for Histogram Matching
if option == "Histogram Matching":
    uploaded_file2 = st.file_uploader("Choose the reference PNG or JPEG file for histogram matching", type=["png", "jpeg"])

# Function to process the image
def apply_algorithm():
    if uploaded_file1 is not None:
        # Convert the uploaded file to an OpenCV image
        image1 = Image.open(uploaded_file1).convert('L')
        image1 = np.array(image1)

        processed_image = None

        if option == "Lowpass Gaussian filter (spatial domain)":
            processed_image = gaussian_filter(image1)
        elif option == "Lowpass Butterworth filter (Frequency domain)":
            processed_image = butterworth_filter(image1)
        elif option == "HighPass Laplacian filter (spatial Domain)":
            processed_image = laplacian_filter(image1)
        elif option == "Histogram Matching" and uploaded_file2 is not None:
            image2 = Image.open(uploaded_file2).convert('L')
            image2 = np.array(image2)
            processed_image = histogram_match(image1, image2)

        # Normalize the processed image to the range [0, 255]
        if processed_image is not None:
            processed_image = np.clip(processed_image, 0, 255)
            processed_image = processed_image.astype(np.uint8)

        # Display spinner in the second column
        with col2:
            with st.spinner("Processing..."):
                time.sleep(1)  # Simulate a delay for 1 second

        # Display the processed image after the spinner
        if processed_image is not None:
            with col2:
                st.image(processed_image, caption='Processed Image', use_column_width=True)
                st.write("Processed Image Statistics:")
                st.write(f"Mean: {np.mean(processed_image):.2f}")
                st.write(f"Standard Deviation: {np.std(processed_image):.2f}")

# If the first file is uploaded, display it
if uploaded_file1 is not None:
    bytes_data1 = uploaded_file1.read()
    col1, col2 = st.columns(2)

    with col1:
        st.image(bytes_data1, caption="First Image", use_column_width=True)
        st.write("Filename:", uploaded_file1.name)

    # Apply button
    st.button(
        "Apply", 
        key=None, 
        help="Apply Algorithm", 
        on_click=apply_algorithm, 
        args=None, 
        type="primary", 
        disabled=(uploaded_file1 is None or (option == "Histogram Matching" and uploaded_file2 is None)),
        use_container_width=True
    )

    # Display the second image if Histogram Matching is selected and the file is uploaded
    if option == "Histogram Matching" and uploaded_file2 is not None:
        bytes_data2 = uploaded_file2.read()
        with col2:
            st.image(bytes_data2, caption="Reference Image", use_column_width=True)
            st.write("Filename:", uploaded_file2.name)
else:
    st.write("Please upload the required image(s).")
