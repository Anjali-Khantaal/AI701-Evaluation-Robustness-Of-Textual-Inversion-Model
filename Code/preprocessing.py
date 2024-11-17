import cv2
import numpy as np
import os
import random
from PIL import Image, ImageEnhance

# Resize the image
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

# Normalize the image to [0, 1]
def normalize_image(image):
    return image / 255.0  # Normalize to [0, 1]

# Add Gaussian Noise with adjustable variance
def add_gaussian_noise(image, severity='low'):
    noise_levels = {'low': 0.01, 'medium': 0.05, 'high': 0.1}
    var = noise_levels.get(severity, 0.01)
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)  # Ensure the pixel values are within [0, 1]

# Add Motion Blur with adjustable kernel size
def add_motion_blur(image, severity='low'):
    kernel_sizes = {'low': 3, 'medium': 5, 'high': 7}
    kernel_size = kernel_sizes.get(severity, 3)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    blurred_image = cv2.filter2D((image * 255).astype(np.uint8), -1, kernel)
    return np.array(blurred_image) / 255.0  # Normalize back to [0, 1]

# Apply Random Color Jitter with adjustable parameters
def color_jitter(image, severity='low'):
    # Define the ranges of jitter based on severity
    jitter_params = {
        'low': (0.1, 0.1, 0.1, 0.05),
        'medium': (0.2, 0.2, 0.2, 0.1),
        'high': (0.3, 0.3, 0.3, 0.15)
    }
    brightness, contrast, saturation, hue = jitter_params.get(severity, (0.1, 0.1, 0.1, 0.05))

    # Convert image from [0, 1] float format to 8-bit integer format (0-255)
    image = Image.fromarray((image * 255).astype(np.uint8))

    # Adjust brightness, contrast, saturation
    image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - brightness, 1 + brightness))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - contrast, 1 + contrast))
    image = ImageEnhance.Color(image).enhance(random.uniform(1 - saturation, 1 + saturation))

    # Apply hue shift by converting to HSV, adjusting, and converting back to RGB
    image = image.convert('HSV')
    data = np.array(image)
    # Here, apply the shift to the hue component directly
    data[..., 0] = (data[..., 0] + int(hue * 255)) % 255  # Apply hue shift
    image = Image.fromarray(data, 'HSV').convert('RGB')

    # Convert back to [0, 1] float format
    return np.array(image) / 255.0  # Normalize back to float [0, 1]


# Add Random Occlusion with adjustable occlusion size
def random_occlusion(image, severity='low'):
    occlusion_sizes = {'low': 0.05, 'medium': 0.15, 'high': 0.3}
    occlusion_size = occlusion_sizes.get(severity, 0.05)
    
    h, w, _ = image.shape
    cutout_height = int(h * occlusion_size)
    cutout_width = int(w * occlusion_size)
    y_start = random.randint(0, h - cutout_height)
    x_start = random.randint(0, w - cutout_width)
    image[y_start:y_start + cutout_height, x_start:x_start + cutout_width] = 0  # Set to black
    return image

# Main preprocessing function that applies each distortion separately
def preprocess_image(image_path, output_size=(512, 512), severity='low'):
    """Preprocess the image by applying augmentations."""
    # Load image
    image = cv2.imread(image_path)
    image = resize_image(image, output_size)
    image = normalize_image(image)

    # Apply augmentations
    image_with_noise = add_gaussian_noise(image, severity)
    image_with_blur = add_motion_blur(image, severity)
    image_with_jitter = color_jitter(image, severity)
    image_with_occlusion = random_occlusion(image, severity)
    
    return image_with_noise, image_with_blur, image_with_jitter, image_with_occlusion

# Process and save images for each severity level
def main(input_folder, output_folder, output_size=(512, 512)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            # Process the image at each severity level (low, medium, high)
            for severity in ['low', 'medium', 'high']:
                processed_images = preprocess_image(image_path, output_size, severity)
                for i, processed_image in enumerate(processed_images):
                    output_filename = f'{severity}_{["noise", "blur", "jitter", "occlusion"][i]}_{filename}'
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, (processed_image * 255).astype(np.uint8))

if __name__ == "__main__":
    input_folder = r".\Training_Dataset"  # CHANGE THIS TO YOUR PATH
    output_folder = os.path.join(input_folder, "Preprocessed")  # Output folder for processed images
    main(input_folder, output_folder)

