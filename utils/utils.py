import numpy as np
import cv2
from IPython.display import clear_output, Image, display
import yaml
import os

# Load the configuration file
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

scale_factor = config['webcam']['resolution'][1]/config['world']['width']

def add_label(image, text):
    label_image = image.copy()
    # Add black background for text
    cv2.rectangle(label_image, (0, 0), (200, 30), (0, 0, 0), -1)
    # Add white text
    cv2.putText(label_image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    return label_image

def distance(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    
def mm_to_pixels(number, scale_factor):
    return int(number * scale_factor)

def pixels_to_mm(number, scale_factor):
    return number / scale_factor

def simulate_robot_movement(position, orientation, command, dt=0.005):
    new_position = np.array(position, dtype=np.float64)
    new_orientation = float(orientation)  
    if command['action'] == 'move_and_rotate':
        # Update orientation
        new_orientation += command['rotation_speed'] * dt
        # Update position based on forward movement in the current orientation
        new_position[0] += command['forward_speed'] * np.cos(new_orientation) * dt  
        new_position[1] += command['forward_speed'] * np.sin(new_orientation) * dt
    
    return new_position, new_orientation

# Draw robot as rectangle
def draw_robot(frame, position, orientation, thymio_size):
    thymio_size = thymio_size.copy()
    position = position.copy()
    thymio_size = np.array([mm_to_pixels(thymio_size['width'], scale_factor), 
                            mm_to_pixels(thymio_size['length'], scale_factor)])
    position = np.array([mm_to_pixels(position[0], scale_factor), 
                         mm_to_pixels(position[1], scale_factor)])
    
    width = thymio_size[0]
    length = thymio_size[1]
    
    # Calculate corner points of rectangle based on center position and orientation
    center = np.array(position, dtype=np.float32)
    
    # Create rotation matrix
    angle = orientation  # orientation should be in radians
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Define corners relative to center (before rotation)
    half_length = length / 2
    half_width = width / 2
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
    front_end = center + rotation_matrix @ np.array([length/2, 0])
    cv2.line(frame, tuple(front_start.astype(int)), tuple(front_end.astype(int)), (255, 255, 0), 2) 

    return frame

def draw_trajectory(frame, position_history):
    if len(position_history) <= 1:
        return frame
        
    path_points = np.array(position_history)
    
    # Draw lines connecting consecutive points
    for i in range(len(path_points) - 1):
        pt1 = tuple(path_points[i].astype(int))
        pt2 = tuple(path_points[i + 1].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    
    for point in path_points:
        cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 255), -1)
            
    return frame

def display_frames(original_frame, processed_frame, trajectory_frame):
    frames = []
    frame_labels = ["Original Frame", "Processed Frame", "Trajectory Frame"]
    input_frames = [original_frame, processed_frame, trajectory_frame]
    
    # Process each frame that exists
    for frame, label in zip(input_frames, frame_labels):
        if frame is not None:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Add label
            labeled_frame = add_label(frame_rgb, label)
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
    
    # Convert to jpg and display
    _, buffer = cv2.imencode('.jpg', combined_frame)
    display(Image(data=buffer.tobytes()))
    clear_output(wait=True)

