from global_nav import GlobalNav
import cv2
import matplotlib.pyplot as plt

def main():
    global_nav = GlobalNav()
    img_path = "frame.jpg"
    img = cv2.imread(img_path)
    contours, obstacles_corners = global_nav.detect_contours(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(contours)
    plt.show()
    
if __name__ == "__main__":
    
    main()