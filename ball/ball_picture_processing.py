import cv2
import numpy as np


# Function loading the image and converting it to the HSV color space
def load_and_convert_to_hsv(image_path):
    # Loads the image
    image = cv2.imread(image_path)
    # Converts to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image, image


# Function creating a mask for the ball's colors
def create_color_mask(hsv_image, lower_bound, upper_bound):
    # Creates a mask with the specified color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return mask


# Function performing morphological operations to remove noise
def apply_morphology(mask, kernel_size):
    # Define the kernel based on the provided size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Apply morphological open and close to remove noise
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    return clean_mask


# Function calculating the center of the object and placing a marker
def mark_object_center(image, mask):
    # Finding contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Gets the largest contour assumed to be the object
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculates the center of the contour
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Places a marker at the center
            cv2.drawMarker(image, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, markerSize=20, thickness=2)
            print('The center of the object is at: (', cx, ',', cy, ')')
            return image, (cx, cy)

    return image, None


# Main processing function
def process_image(image_path):
    # Loads and converts image to HSV
    hsv_image, original_image = load_and_convert_to_hsv(image_path)

    # Ball color range in HSV
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])

    # Creates a mask for the colors
    mask = create_color_mask(hsv_image, lower_bound, upper_bound)

    # Applies morphology to clean the mask
    clean_mask = apply_morphology(mask, kernel_size=7)

    # Marks the center of the object on the original image
    marked_image, center = mark_object_center(original_image.copy(), clean_mask)

    return marked_image, center, clean_mask


# Defines the path to the uploaded image
image_path = 'ball_picture.png'

# Processes the image and retrieves the results
processed_image, center, mask = process_image(image_path)

# Saves the outputs to files
cv2.imwrite('ball_processed_picture.png', processed_image)
cv2.imwrite('ball_mask.png', mask)

# Returns the path to the saved image and the center coordinates if available
processed_image_path = 'ball_processed_picture.png'
mask_image_path = 'ball_mask.png'
center_coordinates = center

processed_image_path, mask_image_path, center_coordinates

