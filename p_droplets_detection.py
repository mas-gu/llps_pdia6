############################################################
#########      Detection of droplets               #########  
#########        in  DIC images                    #########
#########           using openCV                   #########  
#########      Return condensates properties       #########  
#########          in a *.csv file                 #########
############################################################

# %% #001 import module
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import csv

#002 Load the image
image_path = 'xx'
ig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#002 Process the image
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)
img = cv2.convertScaleAbs(ig, alpha=alpha, beta=beta)
blur = cv2.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=50)

#003 Detect circles using HoughCircles
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1., minDist=20,
                           param1=80, param2=20, minRadius=1, maxRadius=20)

# %% 004 Use the detected circles to extract properties
# Check if circles are found
if circles is not None:
    # Convert circle parameters from float to int
    # circles = circles.astype(int)
    circles = np.uint16(np.around(circles))

    # Draw circles on the original image
    img_color = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  # Convert to color image
    for circle in circles[0, :]:
        # Draw the outer circle
        cv2.circle(img_color, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        
        # Display the results
    plt.figure(figsize=(20, 8))
    pdf_filename = image_path.replace("titration_calcium/", "").replace("titration_DIC/", "") + ".pdf"

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(ig, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Blurred image
    plt.subplot(1, 3, 2)
    plt.imshow(blur, cmap='gray')
    plt.title("Blurred Image")
    plt.axis("off")

    # Output with detected circles
    plt.subplot(1, 3, 3)
    plt.imshow(img_color)
    plt.title("Detected Circles")
    plt.axis("off")

    plt.savefig(pdf_filename)
    plt.show()

    
else:
    plt.figure(figsize=(20, 8))
    pdf_filename = image_path.replace("titration_calcium/", "").replace("titration_DIC/", "") + ".pdf"

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # Blurred image
    plt.subplot(1, 2, 2)
    plt.imshow(blur, cmap='gray')
    plt.title("Blurred Image")
    plt.axis("off")

    plt.show()
    plt.savefig(pdf_filename)
    print("No circles detected.")# 
    

#005: Use the detected circles to extract properties
if circles is not None:
    # Number of circles detected
    num_circles = len(circles[0])
    print(f"Number of circles detected: {num_circles}")
    average_radius = np.mean(circles[0, :, 2])
    stdev_radius = np.std(circles[0, :, 2])
    print(f"{average_radius:.2f} {stdev_radius:.2f}")
    
    # Create a CSV file to store circle propertie
    csv_filename = image_path.replace("titration_calcium/", "").replace("titration_DIC/", "") + ".csv"

    
    # Create a CSV file to store circle properties
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["#xx", "Radius", "#Area"])
        for i, circle in enumerate(circles[0, :]):
            radius = circle[2]
            diameter = 2 * radius
            area = 3.1416 * (radius ** 2)
            writer.writerow([i+1,diameter, radius, area])

else:
    print("No properties to extract as no circles were detected.")
    
    
    