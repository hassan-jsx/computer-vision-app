import cv2
import numpy as np

def resize_image(image, max_width=400, max_height=400):
    height, width = image.shape
    if height > max_height or width > max_width:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image

def gaussian_filter(image):
    # Apply Gaussian Blur with a kernel size of 5x5
    return cv2.GaussianBlur(image, (5, 5), 0)

def butterworth_filter(image):
    d = 30
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create Butterworth low-pass filter mask
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = 1 / (1 + (distance / d)**4)
    
    # Apply Fourier Transform
    fshift = np.fft.fftshift(np.fft.fft2(image))
    
    # Apply mask in the frequency domain
    fshift_filtered = fshift * mask
    
    # Inverse Fourier Transform to get back to the spatial domain
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    
    # Convert the result to a real image
    img_back = np.abs(img_back)
    
    # Normalize to the range [0, 255]
    img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back_normalized.astype(np.uint8)

def laplacian_filter(image):
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # Convert back to 8-bit image
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    return laplacian_abs

def histogram_match(image, reference):
    # Compute histograms and cumulative distribution functions (CDF)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    hist_ref, bins_ref = np.histogram(reference.flatten(), 256, [0, 256])
    
    cdf = hist.cumsum()
    cdf_ref = hist_ref.cumsum()
    
    # Normalize the CDFs
    cdf_normalized = cdf * hist_ref.max() / cdf.max()
    cdf_normalized_ref = cdf_ref * hist_ref.max() / cdf_ref.max()
    
    # Interpolate to find the matched CDF values
    cdf_matched = np.interp(cdf_normalized, cdf_normalized_ref, bins_ref[:-1])
    
    # Apply the matched CDF to the input image
    matched_image = np.interp(image.flatten(), bins[:-1], cdf_matched).reshape(image.shape)
    return matched_image.astype(np.uint8)
