import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

#functio to create a rectangle ROI given 2 vertices
def create_rect(x1,y1,x2,y2,edgecolor = 'red'):
    """
    Plots an ROI over a given image based on two vertices.

    Parameters:
        image (2D array): The image to be plotted.
        x1, y1 (int): Coordinates of the first vertex.
        x2, y2 (int): Coordinates of the second vertex.
    """
    # Calculate the width and height of the ROI
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    # Determine the lower-left corner for the ROI
    x_start = min(x1, x2)
    y_start = min(y1, y2)
    
    # Add the ROI as a rectangle
    roi = patches.Rectangle(
        (x_start, y_start), width, height,
        linewidth=2, edgecolor= edgecolor, facecolor='none'
    )
    
    return roi



def plot_image_rect(image_vector, roi):
    img = np.array(image_vector).reshape(100,100)
    x_pred, y_pred, w_pred, h_pred = roi['x'], roi['y'], roi['x'] + roi['w'], roi['y'] + roi['h']
    roi_pred = create_rect(x_pred, y_pred, w_pred, h_pred, 'green')
    
    # Create the plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.add_patch(roi_pred)
    
    # Display the image with the ROI
    
    # Save the plot as an image
    output_path = 'results/output.jpg'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Result saved to {output_path}")
    
    # Show the plot
    plt.show()
    

def save_image_rect(image_vector_list, roi_list):
    for idx, (image_vector, roi) in enumerate(zip(image_vector_list, roi_list)):
        #print(f"Processing item {idx}: ROI = {roi}")
        img = np.array(image_vector).reshape(100,100)
        x_pred, y_pred, w_pred, h_pred = roi['roi_'+str(idx)]['x'], roi['roi_'+str(idx)]['y'], roi['roi_'+str(idx)]['x'] + roi['roi_'+str(idx)]['w'], roi['roi_'+str(idx)]['y'] + roi['roi_'+str(idx)]['h']
        roi_pred = create_rect(x_pred, y_pred, w_pred, h_pred, 'green')
        
        # Create the plot
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.add_patch(roi_pred)
        
        # Display the image with the ROI
        
        # Save the plot as an image
        output_path = 'results/output_{}.jpg'.format(idx)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Result saved to {output_path}")
        