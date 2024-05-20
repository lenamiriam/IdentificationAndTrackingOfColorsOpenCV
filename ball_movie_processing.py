import cv2
import numpy as np

# Function displaying a video from a given path
def display_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Creates a video capture object for the file
    while cap.isOpened():  # Loops until the video ends
        ret, frame = cap.read()  # Reads a frame from the video
        if not ret:  # If frame is not read correctly, breaks the loop
            break
        cv2.imshow('Video', frame)  # Displays the frame in a window
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Waits for 25ms and breaks the loop if 'q' is pressed
            break
    cap.release()  # Releases the video capture object
    cv2.destroyAllWindows()  # Closes all OpenCV windows


# Function converting a frame to HSV color space
def convert_to_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converts the frame to HSV color space
    return hsv_frame


# Function creating a mask based on a color range in HSV space
def create_color_mask(hsv_frame, lower_bound, upper_bound):
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)  # Creates a mask where the specified color range is white
    return mask


# Function applying morphological operations to clean up the mask
def apply_morphology(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Creates a square kernel for morphological operations
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Applies opening to remove noise
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)  # Applies closing to fill holes
    return clean_mask


# Function finding the largest object in the mask and marking its center
def mark_object_center(frame, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Finds contours in the mask
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Finds the largest contour
        M = cv2.moments(largest_contour)  # Calculates moments for the largest contour
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])  # Calculates the x-coordinate of the center
            cy = int(M['m01'] / M['m00'])  # Calculates the y-coordinate of the center
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, markerSize=20, thickness=2)  # Mark the center
    return frame


# Loads the video file into a capture object
video = cv2.VideoCapture('ball_movie.avi')

# Extracts video properties
frame_width = int(video.get(3))  # Gets the width of video frames
frame_height = int(video.get(4))  # Gets the height of video frames
fps = video.get(cv2.CAP_PROP_FPS)  # Gets the frames per second of the video
size = (frame_width, frame_height)  # Creates a size tuple

# Sets up a VideoWriter object to write processed frames to a new file
result = cv2.VideoWriter('ball_processed_movie.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

# Defines the HSV color range for the ball
lower_bound = np.array([0, 100, 100])
upper_bound = np.array([10, 255, 255])

# Processes each frame of the video
while True:
    success, frame = video.read()  # Reads a frame from the video
    if not success:  # If the frame wasn't read correctly, exits the loop
        break

    hsv_frame = convert_to_hsv(frame)  # Converts the frame to HSV color space
    mask = create_color_mask(hsv_frame, lower_bound, upper_bound)  # Creates a mask for the specified color range
    clean_mask = apply_morphology(mask)  # Cleans the mask using morphological operations
    marked_frame = mark_object_center(frame, clean_mask)  # Marks the center of the largest object in the mask

    result.write(marked_frame)  # Writes the processed frame to the output file

# Releases resources
video.release()
result.release()

# Displays the processed video
display_video('ball_processed_movie.avi')
