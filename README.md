# Dragon Under Hand Recognition

## Overview
This Python script enables real-time hand gesture recognition using OpenCV. It dynamically adjusts HSV values and leverages convexity defects to detect gestures and trigger actions like pressing the spacebar.

---

## Features
- **Real-Time Recognition**: Adjust HSV values dynamically for accurate hand detection.
- **ROI-Based Detection**: Optimized performance with a defined detection area.
- **Automated Actions**: Trigger actions (e.g., pressing spacebar) based on gestures.

---

## Requirements
### Software
- Python 3.x
- OpenCV
- NumPy
- PyAutoGUI

### Hardware
- Webcam (built-in or external)

---

## Usage
1. **Save** the script as `gesture_recognition.py`.
2. **Run** the script:
   ```bash
   python gesture_recognition.py
   ```
3. **Adjust HSV Values** using trackbars to fine-tune detection.
4. **Position Hand** inside the green ROI (Region of Interest).
5. **Perform Gestures** like showing 5 fingers to trigger actions (e.g., spacebar press).
6. **Exit** by pressing `q`.

---

## Customization
- **HSV Range**: Modify default HSV values in the script for better accuracy.
- **ROI Dimensions**: Adjust the ROI coordinates `(100, 100, 300, 300)`.
- **Action Commands**: Replace `pyautogui.press('space')` with custom actions.

---

## Troubleshooting
- **No Detection**: Adjust the HSV values using trackbars.
- **Performance Issues**: Minimize background movement and ensure sufficient lighting.
- **Contour Issues**: Ensure the largest contour area exceeds 2000.

---

## Acknowledgements
- **OpenCV**: For robust computer vision capabilities.
- **PyAutoGUI**: For seamless keyboard automation.

---

Simplify gesture control and automate actions with ease using this script!

