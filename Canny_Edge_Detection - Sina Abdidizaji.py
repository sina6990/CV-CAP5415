import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

##############################
### Image Loading Function ###
##############################  
'''This function loads an image from the specified given path, convert it to grayscale and returns it as a NumPy array'''
def image_load(path):
    img = Image.open(path).convert('L')
    return np.array(img)

############################
### Gaussian Mask Kernel ###
############################ 
'''This function creates a 1D Gaussian kernel for a given kernel size and sigma value'''
def gaussian_kernel(size, sigma):
    # creating an array of zeros with the size of the kernel
    kernel = np.zeros(size)
    # finding the center of the kernel by dividing the size by 2
    center = size // 2
    # looping through the kernel size to calculate the Gaussian value for each element in the array
    for i in range(size):
        dist = (i - center)**2
        kernel[i] = np.exp(-dist / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    return kernel

#####################################
### Gaussian Derivate Mask Kernel ###
##################################### 
'''This function creates a 1D Gaussian derivative kernel for a given kernel size and sigma value'''
def gaussian_derivative_kernel(size, sigma):
    # creating an array of zeros with the size of the kernel
    kernel = np.zeros(size)
    # finding the center of the kernel
    center = size // 2
    # looping through the kernel size to calculate the Gaussian derivative value for each element in the array
    for i in range(size):
        dist = i - center
        kernel[i] = (-dist * np.exp(-(dist**2) / (2 * sigma**2))) / (np.sqrt(2 * np.pi) * sigma**3)
    return kernel

##############################
### Convolving over X-axis ###
############################## 
'''This function convolves a given image with a given kernel across the X-axis'''
def convolve_x(image, kernel):
    # Getting the height and width of the image for looping through the image pixels
    image_height, image_width = image.shape
    # Getting the size of the kernel for calculating the padding size
    kernel_size = kernel.shape[0]
    # Calculating the padding size by dividing the kernel size by 2
    padding_size = kernel_size // 2
    # Padding the image with zeros around the borders of the original image
    padded_image = np.pad(image, padding_size, mode='constant', constant_values=0)
    # Creating an empty array of zeros for the output after convolving the image with the kernel
    output = np.zeros_like(image)
    # Looping through the height and width of the image to convolve the image with the kernel
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i+padding_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    
    return output

##############################
### Convolving over Y-axis ###
############################## 
'''This function convolves a given image with a given kernel across the Y-axis'''
def convolve_y(image, kernel):
    # Getting the height and width of the image for looping through the image pixels
    image_height, image_width = image.shape
    # Getting the size of the kernel for calculating the padding size
    kernel_size = kernel.shape[0]
    # Calculating the padding size by dividing the kernel size by 2
    padding_size = kernel_size // 2
    # Padding the image with zeros around the borders of the original image
    padded_image = np.pad(image, padding_size, mode='constant', constant_values=0)
    # Creating an empty array of zeros for the output after convolving the image with the kernel
    output = np.zeros_like(image)
    # Looping through the height and width of the image to convolve the image with the kernel
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_size, j+padding_size]
            output[i, j] = np.sum(region * kernel)
    
    return output

###########################################
### Magnitude and Direction of Gradient ###
###########################################
'''This function computes the magnitude and direction of the gradient given the gradients along the X and Y axes'''
def compute_gradient_magnitude_and_direction(i_x, i_y):
    magnitude = np.sqrt(i_x ** 2 + i_y ** 2)
    direction = np.arctan2(i_y, i_x)
    return magnitude, direction

###############################
### Non-Maximum Suppression ###
###############################
'''This function performs non-maximum suppression to thin out the edges'''
def non_maximum_suppression(magnitude, direction):
    # Get the image dimensions
    image_height, image_width = magnitude.shape
    # Create an empty array for the output
    output = np.zeros_like(magnitude)
    # Convert direction to degrees
    angle = np.rad2deg(direction)
    # Loop through height and width of an image and excluding the border pixels
    for i in range(1, image_height - 1):  
        for j in range(1, image_width - 1):  
            
            # Compare the gradient direction and look at the neighbors' magnitude values
            # left/right neighbors
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                # Right neighbor
                q = magnitude[i, j + 1]
                # Left neighbor  
                r = magnitude[i, j - 1]  
            # bottom-left/top-right diagonal neighbors
            elif 22.5 <= angle[i, j] < 67.5:
                # Bottom-left diagonal neighbor
                q = magnitude[i + 1, j - 1] 
                # Top-right diagonal neighbor
                r = magnitude[i - 1, j + 1] 
            # top/bottom neighbors
            elif 67.5 <= angle[i, j] < 112.5:
                # Bottom neighbor
                q = magnitude[i + 1, j]
                # Top neighbor
                r = magnitude[i - 1, j]
            # top-left/bottom-right diagonal neighbors
            elif 112.5 <= angle[i, j] < 157.5:
                # Top-left diagonal neighbor
                q = magnitude[i - 1, j - 1]  
                # Bottom-right diagonal neighbor
                r = magnitude[i + 1, j + 1] 
            
            # If the magnitude of the current pixel is greater than the magnitudes of the neighbors, keep the value
            # Otherwise, suppress the value and set it to zero
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0
    
    return output


################################
### Hysteresis Thresholding ###
################################
'''This function performs hysteresis thresholding to finalize the edges'''
def hysteresis_thresholding(suppressed, low_threshold, high_threshold):
    # Get the image dimensions
    image_height, image_width = suppressed.shape
    # Create an empty array for the output
    edges = np.zeros_like(suppressed)

    # Mark strong edges (those greater than the high threshold)
    strong_i, strong_j = np.where(suppressed >= high_threshold)
    # Strong edges are marked as 1 (edge)
    edges[strong_i, strong_j] = 1 

    # Mark weak edges (those between the low and high thresholds)
    weak_i, weak_j = np.where((suppressed >= low_threshold) & (suppressed < high_threshold))
    weak_edges = np.zeros_like(suppressed)
    # Weak edges are potential edges
    weak_edges[weak_i, weak_j] = 1  
    # Process weak edges: Check if they are connected to strong edges
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            if weak_edges[i, j] == 1:
                # Check 8-connected neighborhood for strong edges
                if ((edges[i+1, j-1] == 1) or (edges[i+1, j] == 1) or (edges[i+1, j+1] == 1) or
                    (edges[i, j-1] == 1) or (edges[i, j+1] == 1) or
                    (edges[i-1, j-1] == 1) or (edges[i-1, j] == 1) or (edges[i-1, j+1] == 1)):
                    # If any of the neighboring pixels are strong, this weak edge becomes a strong edge
                    edges[i, j] = 1
    
    return edges

# 3096, Plane
# 21077, Cars
# 182053, Bridge

path = "Images/test/3096.jpg"
I = image_load(path)
plt.imshow(I, cmap='gray')

sigma = [1,2,3]
for i in sigma:
    kernel_size = 6 * i + 1
    S_x = convolve_x(I, gaussian_kernel(kernel_size, i))
    plt.imshow(S_x, cmap='gray')
    plt.savefig(f'smoothing_x-axis_{i}.jpg')

    D_x = convolve_x(S_x, gaussian_derivative_kernel(kernel_size, i))
    plt.imshow(D_x, cmap='gray')
    plt.savefig(f'derivative_x-axis_{i}.jpg')

    S_y = convolve_y(I, gaussian_kernel(kernel_size, i))
    plt.imshow(S_y, cmap='gray')
    plt.savefig(f'smoothing_y-axis_{i}.jpg')

    D_y = convolve_y(S_y, gaussian_derivative_kernel(kernel_size, i))
    plt.imshow(D_y, cmap='gray')
    plt.savefig(f'derivative_y-axis_{i}.jpg')

    mag, dir = compute_gradient_magnitude_and_direction(D_x, D_y)
    plt.imshow(mag, cmap='gray')
    plt.savefig(f'gradient_magnitude_{i}.jpg')
    plt.imshow(dir)
    plt.savefig(f'gradient_direction_{i}.jpg')

    suppressed = non_maximum_suppression(mag, dir)
    plt.imshow(suppressed, cmap='gray')
    plt.savefig(f'Non_Maximum_Suppression_{i}.jpg')

    high_threshold = np.percentile(mag, 90) 
    low_threshold = 0.4 * high_threshold
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    plt.imshow(edges, cmap='gray')
    plt.savefig(f'Canny_Edge_Detection_{i}.jpg')