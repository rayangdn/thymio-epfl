import cv2
import matplotlib.pyplot as plt
import numpy as np

from global_nav import GlobalNav

def main():
    global_nav = GlobalNav()
    img_path = "frame.png"
    img = cv2.imread(img_path)
    thymio_goal_positions = {
        "thymio": np.array([30, 30]),
        "goal": np.array([1000, 550])
        }
    trajectory_img, trajectory = global_nav.get_trajectory(img, thymio_goal_positions)
    plt.figure(figsize=(10, 10))
    plt.imshow(trajectory_img)
    plt.show()
    
if __name__ == "__main__":
    
    main()