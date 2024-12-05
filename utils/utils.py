import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output, Image, display


def distance(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def mm_to_pixels(number, scale_factor):
    return int(number * scale_factor)

def pixels_to_mm(number, scale_factor):
    return number / scale_factor

def add_label(image, text):
    label_image = image.copy()
    # Add black background for text
    cv2.rectangle(label_image, (0, 0), (200, 30), (0, 0, 0), -1)
    # Add white text
    cv2.putText(label_image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    return label_image

def display_frames(original_frame=None, processed_frame=None, trajectory_frame=None, save_dir=None):
    frames = []
    frame_labels = ["Original Frame", "Processed Frame", "Trajectory Frame"]
    input_frames = [original_frame, processed_frame, trajectory_frame]
    
    # Process each frame that exists
    for frame, label in zip(input_frames, frame_labels):
        if frame is not None:
            # Convert to RGB
           # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Add label
            labeled_frame = add_label(frame, label)
            frames.append(labeled_frame)
    
    if not frames:  # If no valid frames
        return
    
    # Find the maximum height among all frames
    max_height = max(frame.shape[0] for frame in frames)
    
    # Resize all frames to have the same height while maintaining aspect ratio
    resized_frames = []
    for frame in frames:
        if frame.shape[0] != max_height:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = int(max_height * aspect_ratio)
            resized_frame = cv2.resize(frame, (new_width, max_height))
            resized_frames.append(resized_frame)
        else:
            resized_frames.append(frame)
    
    # Concatenate all frames horizontally
    combined_frame = np.hstack(resized_frames)
    
    # Save the combined frame if save_dir is provided
    if save_dir is not None:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Generate timestamp for unique filename
        # Create full save path with filename
        save_path = os.path.join(save_dir, 'frame.jpg')
        # Save the frame
        cv2.imwrite(save_path, combined_frame)
        
    # Convert to jpg and display
    _, buffer = cv2.imencode('.jpg', combined_frame)
    display(Image(data=buffer.tobytes()))
    clear_output(wait=True)
    
    return combined_frame
def display_processing_steps(steps):
    # Define number of columns and rows
    n_cols = 2
    n_rows = 2
    # Create figure
    plt.figure(figsize=(15, 9))
    
    # Plot each step
    for idx, (title, img) in enumerate(steps.items(), 1):
        plt.subplot(n_rows, n_cols, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    #plt.savefig('./img/vision//obstacles/obstacles_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    clear_output(wait=True)

# Draw robot as rectangle
def draw_robot(frame, position, thymio_width, thymio_length, scale_factor):
    position = position.copy()
    orientation = position[2]
    
    # Convert position and dimensions to pixels
    thymio_width = mm_to_pixels(thymio_width, scale_factor)
    thymio_length = mm_to_pixels(thymio_length, scale_factor) 
    position = np.array([mm_to_pixels(position[0], scale_factor), 
                         mm_to_pixels(position[1], scale_factor)])
    
    # Calculate corner points of rectangle based on center position and orientation
    center = np.array(position, dtype=np.float32)
    
    # Create rotation matrix
    angle = orientation  # orientation should be in radians
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Define corners relative to center (before rotation)
    half_length = thymio_length / 2
    half_width = thymio_width / 2
    corners_rel = np.array([
        [-half_length, -half_width],  # top-left
        [half_length, -half_width],   # top-right
        [half_length, half_width],    # bottom-right
        [-half_length, half_width]    # bottom-left
    ])
    
    # Rotate corners and add center position
    corners = np.array([
        rotation_matrix @ corner + center for corner in corners_rel
    ], dtype=np.int32)
    
    # Draw filled rectangle
    cv2.fillPoly(frame, [corners], (255, 0, 255))
    
    # Draw direction indicator (front of robot)
    front_start = center
    front_end = center + rotation_matrix @ np.array([thymio_length/2, 0])
    cv2.line(frame, tuple(front_start.astype(int)), tuple(front_end.astype(int)), (0, 255, 255), 2) 

    return frame

def draw_trajectory(frame, position_history, scale_factor, color):
    if len(position_history) <= 1:
        return frame
        
    path_points = np.array([[mm_to_pixels(x, scale_factor), mm_to_pixels(y, scale_factor)] for x, y in position_history])
    
    # Draw lines connecting consecutive points
    for i in range(len(path_points) - 1):
        pt1 = tuple(path_points[i].astype(int))
        pt2 = tuple(path_points[i + 1].astype(int))
        cv2.line(frame, pt1, pt2, color, 2)
    
    for point in path_points:
        cv2.circle(frame, tuple(point.astype(int)), 3, color, -1)
            
    return frame 

def update_visualization_frame(trajectory_frame, position_history, filtered_history, 
                             robot_state, width, length, scale_factor):

    if trajectory_frame is None:
        return None
        
    current_frame = trajectory_frame.copy()
    
    # Draw measured trajectory in red
    current_frame = draw_trajectory(current_frame, position_history, 
                                    scale_factor, color=(255, 0, 0))
    
    # Draw filtered trajectory in yellow
    current_frame = draw_trajectory(current_frame, filtered_history, 
                                    scale_factor, color=(255, 255, 0))
    
    # Draw robot using filtered position
    current_frame = draw_robot(current_frame, robot_state[:3], width, 
                               length, scale_factor)
    
    return current_frame
       
def print_status(obstacles_pos, thymio_pos, goal_pos, trajectory):
    # Print obstacles information
    print("\n=== Obstacles Information ===")
    print(f"Number of obstacles detected: {len(obstacles_pos)}")
    print("Obstacle coordinates [mm]:")
    for i, (_, corners) in enumerate(obstacles_pos.items()):
        formatted_corners = "\n    ".join([f"Corner {i+1}: [{x:.1f}, {y:.1f}]" 
                                         for i, (x, y) in enumerate(corners)])
        print(f"\nObstacle {i+1}:\n    {formatted_corners}")

    # Print Thymio information
    print("\n=== Thymio Information ===")
    print(f"Position [mm]: [{thymio_pos[0]:.1f}, {thymio_pos[1]:.1f}]")
    print(f"Orientation [°]: {np.rad2deg(thymio_pos[2]):.1f}")

    # Print Goal information
    print("\n=== Goal Information ===")
    print(f"Position [mm]: [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}]")

    # Print Trajectory information
    print("\n=== Trajectory Information ===")
    print(f"Number of waypoints: {len(trajectory)}")
    print("Waypoint path [mm]:")
    for i, point in enumerate(trajectory):
        print(f"Checkpoint {i+1}: [{point[0]:.1f}, {point[1]:.1f}]")
        