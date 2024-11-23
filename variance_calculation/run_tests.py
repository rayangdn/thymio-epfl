

def measure_static_noise(vision):
    # Place Thymio stationary in a fixed position
    # Record position measurements for ~100 samples
    positions = []
    orientations = []
    
    for _ in range(100):
        # Get camera measurement
        original_frame, _ , pos, orientation, _, found_thymio, _ = vision.get_frame()
        if found_thymio:
            positions.append(pos)
            orientations.append(orientation)
        time.sleep(0.1) 
        cv2.imshow("Frame", original_frame)
    
    positions = np.array(positions)
    orientations = np.array(orientations)
    
    # Calculate variances
    pos_variance = np.var(positions, axis=0)  # [var_x, var_y]
    orientation_variance = np.var(orientations)  # [var_theta]
    
    return pos_variance, orientation_variance

def measure_velocity_noise():
    # Test at different constant velocities (e.g., 0.1, 0.2 m/s)
    target_velocities = [0.1, 0.2]
    velocity_measurements = {v: [] for v in target_velocities}
    
    for target_v in target_velocities:
        # Command Thymio to move straight at constant velocity
        set_thymio_velocity(target_v)  # Replace with your control function
        time.sleep(2)  # Let velocity stabilize
        
        # Record for ~5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            v = get_thymio_velocity()  # Replace with your measurement function
            velocity_measurements[target_v].append(v)
            time.sleep(0.1)
    
    # Calculate variance for each velocity
    velocity_variances = {v: np.var(measurements) 
                         for v, measurements in velocity_measurements.items()}
    
    return velocity_variances

def main():
    # Start the Vision thread
    vision.connect_webcam()
    
    # Measure the static noise
    pos_variance, heading_variance = measure_static_noise(vision)
    
    print("Position variance: ", pos_variance)
    print("Heading variance: ", heading_variance)
    
    # Stop the Vision thread
    vision.cleanup_webcam()
    
if __name__ == "__main__":
    main()