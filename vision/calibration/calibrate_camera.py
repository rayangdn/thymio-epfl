import numpy as np
import cv2
import glob

def calibrate_camera(checkerboard_size=(7,10), square_size=0.015):  # Modified for your pattern
    """
    Calibrate camera using a checkerboard pattern
    
    Args:
        checkerboard_size: Number of inner corners (width, height) - (7,10) for an 8x11 checkerboard
        square_size: Size of a square in meters (0.015 = 15mm)
    Returns:
        camera_matrix: The camera matrix
        dist_coeffs: Distortion coefficients
    """
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # Convert to actual dimensions in meters
    
    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    cap = cv2.VideoCapture(2)  # Use default camera
    
    # Set resolution to 1080p for your Aukey webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    found_count = 0
    required_count = 15  # Number of good patterns to collect
    
    print("Please show the checkerboard pattern from different angles...")
    print(f"Need {required_count} good captures")
    print(f"Checkerboard configuration:")
    print(f"- Physical size: 200mm × 150mm")
    print(f"- Square size: {square_size*1000} mm")
    print(f"- Inner corners: {checkerboard_size[0]}x{checkerboard_size[1]}")
    
    while found_count < required_count:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # Draw and display the corners
        display_frame = frame.copy()
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            # Draw the corners
            cv2.drawChessboardCorners(display_frame, checkerboard_size, corners2, ret)
            
            # On spacebar press, save the points
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                obj_points.append(objp)
                img_points.append(corners2)
                found_count += 1
                print(f"Captured {found_count}/{required_count}")
        
        # Show info on frame
        cv2.putText(display_frame, f"Captured: {found_count}/{required_count}", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display_frame, "Press SPACE when checkerboard is visible", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow('Calibration', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()
    
    if found_count < required_count:
        print("Insufficient patterns captured")
        return None, None
    
    print("Calculating calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], 
                                        camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"\nTotal calibration error: {mean_error/len(obj_points)} pixels")
    
    print("\nCamera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    return camera_matrix, dist_coeffs

def save_calibration(camera_matrix, dist_coeffs, filename="camera_calibration.npz"):
    """Save calibration parameters to a file"""
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibration saved to {filename}")

def load_calibration(filename="camera_calibration.npz"):
    """Load calibration parameters from a file"""
    data = np.load(filename)
    return data['camera_matrix'], data['dist_coeffs']

if __name__ == "__main__":
    # Run calibration with your specific checkerboard parameters
    camera_matrix, dist_coeffs = calibrate_camera(
        checkerboard_size=(7,10),  # 7x10 inner corners for 8x11 squares
        square_size=0.015          # 15mm = 0.015 meters
    )
    
    if camera_matrix is not None:
        # Save the calibration
        save_calibration(camera_matrix, dist_coeffs)