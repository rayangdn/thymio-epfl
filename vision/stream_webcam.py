import cv2
import numpy as np
import cv2.aruco as aruco

# Map ArUco IDs to corner names
CORNER_MAPPING = {
    0: "bottom_left_corner",
    1: "bottom_right_corner",
    2: "top_right_corner",
    3: "top_left_corner",
    4: "tyhmio",
    5: "goal"
}

def connect_webcam(device_id=0):
    """Connect to webcam and start video capture."""
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    return cap

def detect_aruco_markers(frame):
    """
    Detect ArUco markers in the given frame.
    Returns corners and IDs of detected markers.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    parameters = aruco.DetectorParameters()
    
    # Detect markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    
    return corners, ids

def draw_markers(frame, corners, ids):
    """Draw detected markers with their corner names on the frame."""
    if ids is not None:
        # Draw the detected markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Draw the corner name and ID for each marker
        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            center = np.mean(c, axis=0).astype(int)
            
            # Get corner name from mapping, or use ID if not mapped
            corner_name = CORNER_MAPPING.get(marker_id, f"Unknown ID: {marker_id}")
            
            # Draw background rectangle for better text visibility
            text_size = cv2.getTextSize(corner_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, 
                         (center[0] - 5, center[1] - text_size[1] - 5),
                         (center[0] + text_size[0] + 5, center[1] + 5),
                         (0, 0, 0),
                         -1)
            
            # Draw corner name
            cv2.putText(frame, corner_name, 
                       (center[0], center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Try connecting to the webcam
    cap = None
    # Avoid connecting to computer webcam
    for device_id in range(1, 3):
        print(f"Trying to connect to device {device_id}...")
        cap = connect_webcam(device_id)
        if cap is not None:
            print(f"Successfully connected to device {device_id}")
            break
    
    if cap is None:
        print("Could not find webcam on any device ID. Please check connection.")
        return
    
    print("Detected corner mappings:")
    for marker_id, corner_name in CORNER_MAPPING.items():
        print(f"Marker ID {marker_id} -> {corner_name}")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Couldn't read frame.")
                break
            
            # Detect ArUco markers
            corners, ids = detect_aruco_markers(frame)
            
            # Draw markers with corner names
            frame = draw_markers(frame, corners, ids)
            
            # Display the frame
            cv2.imshow('ArUco Corner Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()