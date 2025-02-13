############################################################
#########      Detection of condensates            #########  
#########        in any cell types (*.tif)         #########
#########           using openCV                   #########  
#########      Return ratio of proteins            #########  
#########         IN/OUT the condensates           #########
############################################################

# %% #001 import module 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum

class ParameterSet(Enum):
    SET1 = {
        'alpha': 1.0,
        'celldetection': 0.2, #+ 
        'condensatedetection': 1.75, 
        'd_otsu': 0.7
    }

# Choose which set to use
selected_set = ParameterSet.SET1.value

alpha = selected_set['alpha']
celldetection = selected_set['celldetection']
condensatedetection = selected_set['condensatedetection']
d_otsu = selected_set['d_otsu']

#002 Load the image 
image_path = "data/0009_pink_n.tif" 
# Load the image and convert it to grayscale
def load_and_convert_to_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.convertScaleAbs(image, alpha= alpha, beta=0)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    #grayscale = image
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, grayscale

# Automatically determine a threshold to isolate cells from the background using Otsu's method
def threshold_image(grayscale_image):
    threshold_value, binary_mask = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adjusted_threshold = threshold_value * d_otsu  # Reduce the threshold by 20%
    _, binary_mask = cv2.threshold(grayscale_image, adjusted_threshold, 255, cv2.THRESH_BINARY)   
    return threshold_value, binary_mask

# Separate condensed and dispersed proteins using intensity thresholding within cell areas
# This function processes each cell independently
def separate_proteins_per_cell(grayscale_image, cell_contours):
    results = []
    
    for contour in cell_contours:
        # Create a mask for the current cell
        cell_mask = np.zeros_like(grayscale_image)
        cv2.drawContours(cell_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Mask the grayscale image to include only cell areas
        cell_only_image = cv2.bitwise_and(grayscale_image, grayscale_image, mask=cell_mask)
        
        # Use a fixed threshold to threshold within the cell area
        cell_threshold_value = np.mean(cell_only_image[cell_only_image > 0]) * celldetection #0.2 #adjust for cell detection 
        cell_only_image[cell_only_image < cell_threshold_value] = 0
        
        # Calculate statistics within cell areas
        mean_intensity = np.mean(cell_only_image[cell_only_image > 0])
        condensed_threshold = mean_intensity * condensatedetection  #adjust for condensate detection 
        
        # Create masks for condensed and dispersed proteins based on thresholds
        condensed_mask = cv2.inRange(cell_only_image, condensed_threshold, 255)
        dispersed_mask = cv2.inRange(cell_only_image, mean_intensity, condensed_threshold - 1)
        
        # Calculate intensity ratio
        condensed_intensity = np.sum(grayscale_image[condensed_mask == 255])
        dispersed_intensity = np.sum(grayscale_image[dispersed_mask == 255])
        
        if dispersed_intensity == 0:
            ratio = float('inf')
        else:
            ratio = condensed_intensity / (dispersed_intensity + condensed_intensity)
        
        results.append({
            'contour': contour,
            'condensed_mask': condensed_mask,
            'dispersed_mask': dispersed_mask,
            'intensity_ratio': ratio
        })

    return results


# Apply the masks to create colored layers for visualization
def apply_colored_masks(grayscale_image, condensed_mask, dispersed_mask):
    # Create an RGB image for visualization
    colored_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    
    # Apply the condensed mask (blue color)
    colored_image[np.where(condensed_mask == 255)] = [255, 0, 0]
    
    # Apply the dispersed mask (red color)
    colored_image[np.where(dispersed_mask == 255)] = [0, 0, 255]
    
    return colored_image

# Calculate the ratio of sum of intensities for condensed and dispersed proteins
def calculate_intensity_ratio(grayscale_image, condensed_mask, dispersed_mask):
    condensed_intensity = np.sum(grayscale_image[condensed_mask == 255])
    dispersed_intensity = np.sum(grayscale_image[dispersed_mask == 255])
    
    if dispersed_intensity == 0:
        return float('inf')  # Avoid division by zero
    ratio = condensed_intensity / (dispersed_intensity+condensed_intensity)
    return ratio


# Function to count the number of cells detected and return their contours
def get_cell_contours(cell_mask):
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    # Define minimum and maximum area for cell detection
    min_area = 1500  # Minimum area based on visual inspection
    max_area = 5000000  # Maximum area based on visual inspection
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Fit an ellipse to the contour and check its shape
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (major_axis, minor_axis), angle = ellipse
                aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
                # Ensure the shape is roughly elliptical or circular
                if 0.1 <= aspect_ratio <= 2.0:
                    filtered_contours.append(contour)
    
    number_of_cells = len(filtered_contours)
    print(f"n {number_of_cells}")
    return filtered_contours
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    number_of_cells = len(contours)
    print(f"n: {number_of_cells}")
    return contours

# Function to display the image with cell contours and cell numbering
def display_cells_with_contours(original_image, cell_contours):
    display_image = original_image.copy()
    for idx, contour in enumerate(cell_contours):
        cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 2)
        # Calculate the center of the cell to place the number
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            cv2.putText(display_image, str(idx + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return display_image


def plot_intensity_histogram(grayscale_image, output_path=None):
    """Create and save a separate intensity histogram plot"""
    plt.figure(figsize=(8, 6))
    plt.hist(grayscale_image.ravel(), bins=256, range=[0,256], density=True)
    plt.xlim(-5, 50)
    plt.title("Intensity Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_cell_intensity_histogram(grayscale_image, cell_mask, output_path=None):
    """Histogram of intensities ONLY within detected cells"""
    # Mask the image to keep only cell regions
    cell_pixels = cv2.bitwise_and(grayscale_image, grayscale_image, mask=cell_mask)
    plt.figure(figsize=(8, 6))
    plt.hist(cell_pixels[cell_pixels > 0].ravel(), bins=256, range=[0,256], density=True)
    plt.xlim(80, 250)
    plt.title("Intensity Distribution (Cell Regions Only)")
    plt.yscale('log')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


# Main function to execute all steps
def main(image_path):
    # Load and process the image
    original_image, grayscale_image = load_and_convert_to_grayscale(image_path)

    # Threshold to separate cells from the background
    threshold_value, cell_mask = threshold_image(grayscale_image)

    # Get the cell contours
    cell_contours = get_cell_contours(cell_mask)

    # Display the image with cell contours and cell numbering
    display_cells_with_contours(original_image, cell_contours)
    
    # Generate display image with contours and numbering
    display_image = display_cells_with_contours(original_image, cell_contours)
    
    # Separate condensed and dispersed proteins within each cell area
    cell_analysis_results = separate_proteins_per_cell(grayscale_image, cell_contours)
   
    # Create visualization with colored masks
    final_colored_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    for result in cell_analysis_results:
        condensed_mask = result['condensed_mask']
        dispersed_mask = result['dispersed_mask']
        ratio = result['intensity_ratio']

        # Apply the condensed mask (blue color)
        final_colored_image[np.where(condensed_mask == 255)] = [255, 0, 0]
        
        # Apply the dispersed mask (red color)
        final_colored_image[np.where(dispersed_mask == 255)] = [0, 0, 255]
        
        # Optionally, draw a contour around each cell
        cv2.drawContours(final_colored_image, [result['contour']], -1, (0, 255, 0), 2)

        # Print the ratio for each cell
        
        print(f"{ratio:.2f}")

    # Calculate intensity ratio
    ratio = calculate_intensity_ratio(grayscale_image, condensed_mask, dispersed_mask)
    
    # Display original and masked images side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Panel 2: 
    ax2.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Cell Detection")
    
    # Panel 2: 
    ax3.imshow(cv2.cvtColor(final_colored_image, cv2.COLOR_BGR2RGB))
    ax3.set_title("Condensed (Blue) vs Dispersed (Red) Proteins with Cell Contours (Green)")
    ax3.axis('off')
    
    fig_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.pdf"
    plt.savefig(fig_filename)
    
    plt.show()
    
    # New histogram plot generation
    #hist_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_histogram.pdf"
    plot_cell_intensity_histogram(grayscale_image, cell_mask, "cell_histogram.pdf")
    #plot_intensity_histogram(grayscale_image, hist_filename)

# Example usage
if __name__ == "__main__":
    main(image_path)
