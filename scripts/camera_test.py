import cv2

# Open the default webcam (0)
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
