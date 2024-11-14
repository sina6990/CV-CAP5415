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
    plt.savefig(os.path.join(current_directory, f'Results/Binarization/Historgram_intensity_image_{number+1}.jpg'))

# Binarize the image using a given threshold value
def binarize_image(image, threshold):
    '''
    Arguments:
         image: Grayscale image as a NumPy array
         threshold: Threshold value for binarization
    Returns:
         Image: Binarized image as a PIL Image
    '''
    binary_image = np.where(image > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_image)

# Main function to load and process three different images
def main():
    '''
    It loads images, allows the user to select thresholds, and saves binarized images
    '''
    image_list = ['image_1.jpg', 'image_2.jpg', 'image_3.jpg']
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepaths = [os.path.join(current_directory, image) for image in image_list]
    thresholds = {'image_1.jpg': 150, 'image_2.jpg': 100, 'image_3.jpg': 120}
    
    for i, filepath in enumerate(filepaths):
            image = load_image(filepath)
            plot_histogram(image, i, current_directory)
            # Get the treshold value for the image
            threshold = thresholds[image_list[i]]
            # Perform binarization using the decided threshold
            binary_image = binarize_image(image, threshold)
            binary_image.save(os.path.join(current_directory, f'Results/Binarization/Binarizied_image_{i+1}.jpg'))

if __name__ == "__main__":
    main()
