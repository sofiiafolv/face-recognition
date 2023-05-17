import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_m = 192
img_n = 168


def draw_image(path):
    global img_m
    global img_n
    with open(path, "rb") as pgmf:
        im = plt.imread(pgmf)
    face_col = np.ndarray.flatten(im)
    face_col = np.reshape(face_col, (-1, 1))
    img_m = im.shape[0]
    img_n = im.shape[1]
    img = plt.imshow(im)
    img.set_cmap("gray")
    plt.axis("off")
    plt.show()


def get_column_from_pgm(photo_path):
    with open(photo_path, "rb") as pgmf:
        im = plt.imread(pgmf)
    face_column = np.ndarray.flatten(im)
    return np.reshape(face_column, (-1, 1))


def get_faces_matrix(dir_with_faces):
    # files = os.listdir(dir_with_faces)
    result = np.zeros(img_m * img_n)
    filenames = []
    for root, dirs, files in os.walk(dir_with_faces):
        for filename in files:
            original_photo_path = os.path.join(root, filename)
            face_column = get_column_from_pgm(original_photo_path)
            result = np.column_stack((result, face_column))
            filenames.append(filename)
    return result[:, 1:], filenames


def convert_to_pgm(image_path, output_path):
    image = Image.open(image_path)
    resized_image = image.resize((img_m, img_n))
    grayscale_image = resized_image.convert("L")
    grayscale_image.save(output_path)
    print(f"Image {image_path} converted and saved as {output_path}")


def get_column_from_user_file(path):
    new_extension = ".pgm"
    file_name, _ = os.path.splitext(path)
    pgm_image_path = os.path.join(
        "./training_faces", file_name + new_extension)
    convert_to_pgm(path, pgm_image_path)
    return get_column_from_pgm(pgm_image_path)


def proccess_user_file_into_column(photo_path):
    im = Image.open(photo_path)
    bw_image = im.convert("L")
    im = bw_image.resize((img_m, img_n))
    matrix = np.array(im)
    face_col = np.ndarray.flatten(matrix)
    face_col = np.reshape(face_col, (-1, 1))
    return face_col


def get_files_in_directory(relative_path):
    current_directory = os.getcwd()
    directory_path = os.path.abspath(
        os.path.join(current_directory, relative_path))
    file_list = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list
