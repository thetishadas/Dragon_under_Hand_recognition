#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pyautogui
import math

# Function to do nothing (used for trackbars)
def nothing(x):
    pass

# Create trackbars for dynamically adjusting HSV values
cv2.namedWindow("HSV Adjustments")
cv2.createTrackbar("LH", "HSV Adjustments", 0, 179, nothing)  # Lower Hue
cv2.createTrackbar("LS", "HSV Adjustments", 30, 255, nothing)  # Lower Saturation
cv2.createTrackbar("LV", "HSV Adjustments", 60, 255, nothing)  # Lower Value
cv2.createTrackbar("UH", "HSV Adjustments", 20, 179, nothing)  # Upper Hue
cv2.createTrackbar("US", "HSV Adjustments", 255, 255, nothing)  # Upper Saturation
cv2.createTrackbar("UV", "HSV Adjustments", 255, 255, nothing)  # Upper Value

# Open the camera
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Define a region of interest (ROI)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian Blur for smoothing
    blurred = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Convert the ROI to HSV color-space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars
    lh = cv2.getTrackbarPos("LH", "HSV Adjustments")
    ls = cv2.getTrackbarPos("LS", "HSV Adjustments")
    lv = cv2.getTrackbarPos("LV", "HSV Adjustments")
    uh = cv2.getTrackbarPos("UH", "HSV Adjustments")
    us = cv2.getTrackbarPos("US", "HSV Adjustments")
    uv = cv2.getTrackbarPos("UV", "HSV Adjustments")

    # Create a binary mask for the given HSV range
    mask = cv2.inRange(hsv, np.array([lh, ls, lv]), np.array([uh, us, uv]))

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Select the largest contour assuming it's the hand
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 2000:
            raise ValueError("Contour area too small.")

        # Draw bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Compute convex hull and defects
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)

        # Initialize defect count
        defect_count = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(largest_contour[s][0])
            end = tuple(largest_contour[e][0])
            far = tuple(largest_contour[f][0])

            # Compute distances and angle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 180) / 3.14

            # If the angle is less than 90, consider it a finger
            if angle <= 90:
                defect_count += 1
                cv2.circle(crop_image, far, 5, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Perform an action based on defect count
        if defect_count >= 4:
            pyautogui.press('space')
            cv2.putText(frame, "JUMP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        # No significant contour or errors
        cv2.putText(frame, "Adjust HSV or Position", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display outputs
    cv2.imshow("Gesture", frame)
    cv2.imshow("Mask", mask)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()

