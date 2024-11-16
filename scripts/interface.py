import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

class Interface:
    def __init__(self):
        # Load config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)             
        config_path = os.path.join(parent_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
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
        
        self.scale_factor = self.config['webcam']['resolution'][1] / self.config['world']['width']
        
        # Initialize plots as None
        self.plots = [None for _ in range(4)]
        
        print("Interface Initialized")

    def update_display(self, original_frame, process_frame, trajectory_frame):
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
        
        plt.draw()
        plt.pause(0.001)

    def is_window_open(self):
        return bool(plt.get_fignums())

    def cleanup(self):
        plt.close('all')