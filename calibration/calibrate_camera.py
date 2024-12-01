import numpy as np
import cv2
import os

def calibrate_camera(checkerboard_size=(7,10), square_size=0.015): 
    # Create save directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(parent_dir, 'img', 'calibration')
    os.makedirs(save_dir, exist_ok=True)

    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # Convert to actual dimensions in meters
    
    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    cap = cv2.VideoCapture(2)  
    
    # Set resolution to 1080p 
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
            
            # On spacebar press, save the points and the image
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                obj_points.append(objp)
                img_points.append(corners2)
                found_count += 1
                
                # Save the image
                img_filename = os.path.join(save_dir, f'calibration_img_{found_count:02d}.png')
                cv2.imwrite(img_filename, frame)
                print(f"Captured {found_count}/{required_count} - Saved as {img_filename}")
        
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

    # Capture and save comparison images
    print("\nCapturing comparison images...")
    cap = cv2.VideoCapture(2)  # Reopen the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Create undistorted version of the frame
        h, w = frame.shape[:2]
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

        # Display both frames
        cv2.imshow('Original', frame)
        cv2.imshow('Calibrated', undistorted)
        cv2.putText(frame, "Press 'S' to save both images, 'Q' to quit", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save both images
            original_path = os.path.join(save_dir, 'original_comparison.png')
            calibrated_path = os.path.join(save_dir, 'calibrated_comparison.png')
            
            cv2.imwrite(original_path, frame)
            cv2.imwrite(calibrated_path, undistorted)
            print(f"Saved comparison images:")
            print(f"Original: {original_path}")
            print(f"Calibrated: {calibrated_path}")
            break
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    
    return camera_matrix, dist_coeffs

def save_calibration(camera_matrix, dist_coeffs, filename="camera_calibration.npz"):
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibration saved to {filename}")

def load_calibration(filename="camera_calibration.npz"):
    data = np.load(filename)
    return data['camera_matrix'], data['dist_coeffs']

if __name__ == "__main__":
    # Run calibration with the checkerboard parameters
    camera_matrix, dist_coeffs = calibrate_camera(
        checkerboard_size=(7,10),  # 7x10 inner corners for 8x11 squares
        square_size=0.015          # 15mm = 0.015 meters
    )
    
    if camera_matrix is not None:
        # Save the calibration
        save_calibration(camera_matrix, dist_coeffs)