import matplotlib.pyplot as plt
import numpy as np
from utils.utils import mm_to_pixels

class Interface:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 10))
        gs = self.fig.add_gridspec(2, 2)
        gs.update(left=0.01, right=0.99, bottom=0.05, top=0.95, wspace=0.02, hspace=0.15)
        
        # Create subplots
        self.axes = [
            self.fig.add_subplot(gs[0, 0]),  # ax1
            self.fig.add_subplot(gs[0, 1]),  # ax2
            self.fig.add_subplot(gs[1, 0]),  # ax3
            self.fig.add_subplot(gs[1, 1])   # ax4
        ]
        
        self.fig.canvas.manager.set_window_title('Vision Interface')
        
        # Initialize plots as None and add robot artists
        self.plots = [None for _ in range(4)]
        self.robot_body = None
        self.robot_direction = None
        
        print("Interface Initialized")
    
    def draw_robot(self, position, orientation, size=10):
        pos_px = mm_to_pixels(position)
        
        # Clear previous robot visualization if it exists
        if self.robot_body is not None:
            self.robot_body.remove()
        if self.robot_direction is not None:
            self.robot_direction.remove()
        
        # Draw robot body as a circle
        self.robot_body = plt.Circle((pos_px[0], pos_px[1]), size, color='magenta', fill=True)
        self.axes[2].add_artist(self.robot_body)
        
        # Draw orientation line
        end_x = pos_px[0] + size * 1.5 * np.cos(orientation)
        end_y = pos_px[1] + size * 1.5 * np.sin(orientation)
        self.robot_direction = self.axes[2].plot([pos_px[0], end_x], [pos_px[1], end_y], 'r-', linewidth=2)[0]
        
        # Update display
        self.fig.canvas.draw_idle()
    
    def update_display(self, original_frame, process_frame, trajectory_frame, current_position=None, current_orientation=None):
        frames = [original_frame, process_frame, trajectory_frame, original_frame]
        titles = ['Webcam View', 'Processing View', 'Trajectory View', 'Result View']
        
        for plot_idx, (frame, title, ax) in enumerate(zip(frames, titles, self.axes)):
            if frame is None:
                continue
            # Update or create image plot
            if self.plots[plot_idx] is None:
                self.plots[plot_idx] = ax.imshow(frame)
                ax.set_title(title, fontsize=12, pad=10)
                ax.axis('off')
            else:
                self.plots[plot_idx].set_data(frame)
        
        # If position and orientation are provided, update robot visualization
        if current_position is not None and current_orientation is not None:
            self.draw_robot(current_position, current_orientation)
        
        plt.draw()
        plt.pause(0.01)

    def is_window_open(self):
        return bool(plt.get_fignums())

    def cleanup(self):
        plt.close('all')