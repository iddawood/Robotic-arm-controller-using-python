import cv2
import numpy as np
import serial
import time

# Initialize serial connection to the robotic arm (adjust COM port and baud rate)
arm = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Allow time for the connection to initialize

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to HSV for color-based object detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color range for the object (e.g., detecting a blue object)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask for the blue object
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour and its center
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Draw contour and center point
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            # Send position to the robotic arm (example command format)
            command = f'GOTO {cx},{cy}\n'
            arm.write(command.encode())
            print(f"Command sent: {command}")
    
    # Show the output frame
    cv2.imshow('Frame', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
arm.close()
