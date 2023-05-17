import numpy as np
import matplotlib.pyplot as plt
from preprocessing_data import *

for j in range(15, 40):
    image_matrix = get_faces_matrix('./CroppedYale/yaleB'+str(j))

    num_per_row = 8
    num_per_col = 8

    fig, axs = plt.subplots(num_per_col, num_per_row, figsize=(20, 20))

    if num_per_col == 1 and num_per_row == 1:
        axs = np.array([axs])

    for i in range(image_matrix.shape[1]):
        ax = axs[i // num_per_row, i % num_per_row]
        image = image_matrix[:, i].reshape((192, 168))
        ax.imshow(image, cmap='gray')
        ax.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    if num_per_col * num_per_row > image_matrix.shape[1]:
        for i in range(image_matrix.shape[1], num_per_col * num_per_row):
            axs.flat[i].set_visible(False)
    plt.show()
    plt.savefig("./photos_per_person/all_photos_person"+str(j)+".png")
    plt.close()

import os

def keep_first_elements(directory):
    """
    Iterates through each directory within the specified directory and keeps only the first 35 elements.

    Args:
        directory (str): The directory path.

    Returns:
        None
    """
    # Iterate over directories within the main directory
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)

            # Get all items in the directory
            items = os.listdir(dir_path)

            # Sort the items and keep only the first 35
            sorted_items = sorted(items)
            items_to_keep = sorted_items[35:]

            # Iterate over the items and delete the rest
            for item in sorted_items:
                item_path = os.path.join(dir_path, item)
                if item not in items_to_keep:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        os.rmdir(item_path)

# Provide the path to the main directory
main_directory = '/home/beheni/UCU/second_term/LA/face-recognition/testing_faces'

# Call the function to keep the first 35 elements in each directory
keep_first_elements(main_directory)