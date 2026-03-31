import numpy as np
import cv2

# Define object-specific variables  
dist = 0
focal = 450
pixels = 30
width = 4

# Function to calculate distance from the camera
def get_dist(rectangle_params, image):
    pixels = rectangle_params[1][0]  # Width of the detected object in pixels
    print(f"Detected Object Width in Pixels: {pixels}")

    # Calculate distance
    dist = (width * focal) / pixels

    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 30)  
    fontScale = 0.6 
    color = (0, 0, 255) 
    thickness = 2

    # Display distance on the image
    image = cv2.putText(image, 'Distance from Camera (cm):', org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, f"{dist:.2f} cm", (10, 60), font,  
                        fontScale, color, thickness, cv2.LINE_AA)

    return image

# Initialize video capture
cap = cv2.VideoCapture(0)

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')

cv2.namedWindow('Object Distance Measurement', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Distance Measurement', 700, 600)

# Main loop to process video frames
while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Predefined mask for green color detection
    lower = np.array([37, 51, 24])
    upper = np.array([83, 104, 131])
    mask = cv2.inRange(hsv_img, lower, upper)

    # Remove extra noise
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find contours
    cont, _ = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:1]  # Keep the largest contour

    for cnt in cont:
        # Check if contour area is within a valid range
        if 100 < cv2.contourArea(cnt) < 306000:
            rect = cv2.minAreaRect(cnt)  # Get the minimum area rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, (255, 0, 0), 3)

            # Calculate and display distance
            img = get_dist(rect, img)

    cv2.imshow('Object Distance Measurement', img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
