############################################################
#########      Detection of condensates            #########  
#########        in any cell types (*.tif)         #########
#########           using openCV                   #########  
#########      Return condensates properties       #########  
#########          in a *.csv file                 #########
############################################################

# %% #001 import module
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from enum import Enum

#set parameters for condensates detection
class ParameterSet(Enum):
    SET1 = {
        'alpha': 1,
        'para1': 25, #+ open / - close morphology
        'para2': -7, #granular separation
        'ker1': 3,
        'ker2': 3,
        'min_circularity': 0.2,
        'max_circularity': 1,
        'perce': 10,
    }

# Choose which set to use
selected_set = ParameterSet.SET1.value

alpha = selected_set['alpha']
para1 = selected_set['para1']
para2 = selected_set['para2']
ker1 = selected_set['ker1']
ker2 = selected_set['ker2']
min_circularity = selected_set['min_circularity']
max_circularity = selected_set['max_circularity']
perce = selected_set['perce']

#002 Load the image
image_path = "x"

image_p = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# %%
#003 Adjust contrast of the image
beta = 0   # Brightness control (0-100)
contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

#004 define the scale 
# Define the known length of the scale bar in micrometers (e.g., 10 μm)
scale_bar_length_pixels = 112
known_scale_length_um = 10  # Adjust this value based on the actual scale length in the image

scale = known_scale_length_um / scale_bar_length_pixels

# Exclude the scale region from the image to avoid detecting it as a shape
image_without_scale = contrast_image.copy()
image_without_scale[-100:, -200:] = 0  # Set the scale region to black to exclude it from detection

# %%
##005 Apply morphological operations to separate connected components
## block size: The size of the neighborhood (block size) used to calculate the threshold. It must be an odd number.
## Constant: A constant subtracted from the calculated threshold. It fine-tunes the threshold value.

def preprocess_image(image):
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, para1, para2)
    kernel = np.ones((ker1, ker2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened

opened = preprocess_image(image_without_scale)

def plot_morphology():
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the adaptive thresholding result
    axes[0].imshow(contrast_image, cmap='gray')
    axes[0].set_title('contrast_image')
    axes[0].axis('off')  # Hide axes

    # Display the result after morphological operations
    axes[1].imshow(opened, cmap='gray')
    axes[1].set_title('After Morphological Opening')
    axes[1].axis('off')  # Hide axes

    # Show the plots
    plt.show()
    
#plot = plot_morphology()

# %%
#006 Detect contours
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def plot_contour():
    # Convert the grayscale image to BGR for color drawing
    colored_image = cv2.cvtColor(image_without_scale, cv2.COLOR_GRAY2BGR)

    # Draw the contours on the image in green
    cv2.drawContours(colored_image, contours, -1, (0, 255, 0), 1)

    # Display the image with contours
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Contours')
    plt.axis('off')
    plt.show()

#007 Draw contours and calculate properties

output_image = cv2.cvtColor(contrast_image, cv2.COLOR_GRAY2BGR)  # Overlay the detected shapes on the original image

# Calculate properties of all contours (only radius)
radii = []
circu = [] 

for contour in contours:
    if len(contour) >= 6:
        ellipse = cv2.fitEllipse(contour)
        (x, y), (major_axis, minor_axis), angle = ellipse
        radius = (major_axis / 2) * scale      
        radii.append(radius)

# fixed thresholds for radius
#min_radius = 0.8 # Minimum radius in μm #0.3
#max_radius = 1.5  # Maximum radius in μm 

def radiiiiii():
    # Plot histogram of radii
    plt.hist(radii, bins= 30, edgecolor='black')
    plt.title('Histogram of Radii')
    plt.xlabel('Radius (μm)')
    plt.ylabel('Frequency')
    plt.show()

rad= radiiiiii()

# Calculate dynamic thresholds for radius
min_radius = np.percentile(radii, perce)  # 10th percentile
max_radius = np.percentile(radii, 100)  # 90th percentile

ellipse_count = 0

#Prepare CSV file to write the properties
csv_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Area (μm^2)", "Max Radius (μm)", "Min Radius (μm)"])  # Updated header

    # Filter contours using dynamic radius thresholds
    for contour in contours:
        if len(contour) >= 6:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis, minor_axis), angle = ellipse
            radius = (major_axis / 2) * scale
            circularity = (4 * np.pi * cv2.contourArea(contour)) / (cv2.arcLength(contour, True) ** 2)
            
            if min_radius <= radius <= max_radius and min_circularity <= circularity <= max_circularity:
                # Draw the ellipse on the output image
                cv2.ellipse(output_image, ellipse, (0, 255, 0), 1)

                # Calculate properties of the ellipse
                #number= 
                max_r = (major_axis /2 ) * scale
                min_r = (minor_axis /2 ) * scale
                max_diameter = major_axis * scale
                min_diameter = minor_axis * scale
                area = np.pi * (max_diameter / 2) * (min_diameter / 2)
                ellipse_count += 1
                # Write the calculated properties to the CSV file
                writer.writerow([f"{area:.2f}", f"{min_r:.2f}", f"{max_r:.2f}"])
           
    print("Number of ellispe", ellipse_count)

# Draw a rectangle around the scale region in the output image
start_point = (output_image.shape[1] - 200, output_image.shape[0] - 100)
end_point = (output_image.shape[1], output_image.shape[0])
color = (0, 0, 255)  # Red color for the rectangle
thickness = 2
cv2.rectangle(output_image, start_point, end_point, color, thickness)

# Create the legend text with parameters
legend_text = (
    f"alpha: {alpha}, beta: {beta}\n"
    f"min_radius: {min_radius} μm, max_radius: {max_radius} μm\n"
)

# Display the result
plt.figure(figsize=(10, 10))
plt.figtext(0.5, 0.01, legend_text, wrap=True, horizontalalignment='center', fontsize=10)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Ovals with Scale Region Highlighted")
fig_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.pdf"
plt.savefig(fig_filename)
plt.axis("off")
plt.show()

# %% all in 
def plot_processing_steps():
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Wider figure for 3 columns
    plt.tight_layout(pad=0)  # Add spacing between subplots

    # Original image with contrast adjustment
    axes[0].imshow(image_p, cmap='gray')
    axes[0].set_title('original')
    axes[0].axis('off')

    # Morphological processing result
    axes[1].imshow(opened, cmap='gray')
    axes[1].set_title('Morphological Operations\n(Kernel={}x{}, Params={},{})'.format(ker1, ker2, para1, para2))
    axes[1].axis('off')

    # Final detection result
    axes[2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Detected Structures\n(Count={}, MinR={:.1f}μm, MaxR={:.1f}μm)'.format(
        ellipse_count, min_radius, max_radius))
    axes[2].axis('off')

    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.pdf"
    plt.savefig(output_filename, 
                format='pdf', 
                bbox_inches='tight', 
                dpi=300, 
                pad_inches=0.1)
    
    plt.show()

plot_processing_steps()  # Removed unnecessary assignment
