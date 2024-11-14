from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Load an image and convert it to grayscale
def load_image(filepath):
    '''
    Arguments: Path to the image file
    Returns: Grayscale image as a NumPy array
    '''
    try:
        image = Image.open(filepath).convert("L")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image at '{filepath}' could not be loaded.")
    return np.array(image)

# Plot the histogram for an image to visualize pixel intensity distribution
def plot_histogram(image, number, current_directory):
    '''
    Arguments:
         image: Grayscale image as a NumPy array
         title: Title for the histogram plot
    Returns:
        Saves the distribution plot of pixel intensities
    '''
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Pixel Intensities for image {number+1}')
    plt.savefig(os.path.join(current_directory, f'Results/Otsu/Historgram_intensity_image_{number+1}.jpg'))

def otsu_thresholding(image):
    # Calculate the histogram of pixel intensities
    histogram, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 256))
    
    # Calculate the probability of each intensity level
    total_pixels = image.shape[0] * image.shape[1]
    pixel_probs = histogram / total_pixels 
    
    # Calculate cumulative sums for weights and cumulative mean for each threshold
    cumulative_sum = np.cumsum(pixel_probs)
    cumulative_mean = np.cumsum(pixel_probs * np.arange(256))

    # Calculate the global mean (This is the mean intensity of the entire image)
    global_mean = cumulative_mean[-1]

    # Initialize between-class variance for each threshold
    between_class_variance = np.zeros(256)

    # Calculate between-class variance for each possible threshold
    for i in range(256):
        weight_background = cumulative_sum[i]          # Weight of the background
        weight_foreground = 1 - weight_background      # Weight of the foreground
        
        if weight_background == 0 or weight_foreground == 0:
            continue  # Skip if background or foreground is empty
        
        mean_background = cumulative_mean[i] / weight_background                  # Mean of the background
        mean_foreground = (global_mean - cumulative_mean[i]) / weight_foreground  # Mean of the foreground
        
        # Compute the between-class variance
        between_class_variance[i] = weight_background * weight_foreground * np.power(mean_background - mean_foreground, 2)
    
    # Find the threshold with the maximum between-class variance
    optimal_threshold = np.argmax(between_class_variance)
    
    # Step 9: Apply the threshold to segment the image
    thresholded_image = np.where(image >= optimal_threshold, 255, 0)
    
    return thresholded_image, optimal_threshold

def main():
    image_list = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg']  
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepaths = [os.path.join(current_directory, image) for image in image_list]
    
    for i, filepath in enumerate(filepaths):
        image = load_image(filepath)
        plot_histogram(image, i, current_directory)
        
        # Perform Otsu's thresholding
        otsu_image, threshold = otsu_thresholding(image)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Image {i+1}')
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f'Otsu Thresholding (Threshold: {threshold})')
        plt.imshow(otsu_image, cmap='gray')
        plt.axis('off')

        plt.savefig(os.path.join(current_directory, f'Results/Otsu/Otsu_image_{i+1}.jpg'))

if __name__ == "__main__":
    main()
