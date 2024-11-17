import cv2
import sys

def start_camera_stream():
    # Initialize video capture from default camera (usually 0)
    cap = cv2.VideoCapture(2)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        sys.exit()
    
    # Set frame dimensions (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Display the frame
            cv2.imshow('Camera Stream', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release everything when job is finished
        cap.release()
        cv2.destroyAllWindows()
