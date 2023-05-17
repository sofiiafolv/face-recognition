import numpy as np

from preprocessing_data import *
from svd import *
from svd_qr import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="path to the photo to be recognized")
    parser.add_argument("k", type=int, help="number k for best rank k approximation")
    parser.add_argument(
        "svd",
        type=int,
        choices=[0, 1, 2],
        help="0 - svd using power method, 1 - svd using library functions for EV's and EVc's, 2 - svd from numpy",
    )
    args = parser.parse_args()
    return args.path, args.k, args.svd


def count_files(relative_path):
    current_directory = os.getcwd()
    directory_path = os.path.abspath(os.path.join(current_directory, relative_path))
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)

    return file_count


def main():
    # parse arguments
    input_path, k, svd_type = parse_args()

    # form a matrix with all photos
    training_faces, filenames = get_faces_matrix("../training_faces")
    number_files = count_files("../training_faces")
    print(count_files)

    # find average face and deduct it from all photos
    average_face = np.mean(training_faces, axis=1)
    normalized_faces = (
        training_faces - np.tile(average_face, (training_faces.shape[1], 1)).T
    )

    # calculate SVD
    if svd_type == 0:
        U, sigma, VT = reduced_svd_using_qr(normalized_faces, k)
    elif svd_type == 1:
        U, sigma, VT = k_rank_approximation_from_scratch(normalized_faces, k)
    else:
        U, sigma, VT = np.linalg.svd(normalized_faces, full_matrices=False)
        U, sigma, VT = U[:, :k], sigma[:k], VT[:k, :]
    # project input image onto eigenfaces space
    test_face_norm = np.squeeze(get_column_from_pgm(input_path)) - average_face
    projected_face_coord = U.T @ test_face_norm
    projected_face_coord = projected_face_coord/np.linalg.norm(projected_face_coord)
    # calculate sigma @ VT to easily find coordinate vectors of training set images in the eigenfaces space
    sigma_VT = np.diag(sigma) @ VT

    # calculate distances between coordinate vectors of input face and training set images in the eigenfaces space
    recognized_faces = []
    for i in range(number_files):
        dist = np.linalg.norm(projected_face_coord - sigma_VT[:, i]/np.linalg.norm(sigma_VT[:, i]))
        if dist < 4000:
            recognized_faces.append((dist, i))

    if  recognized_faces:
        recognized_faces.sort()
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        img_avg = ax1.imshow(
            np.reshape(test_face_norm + average_face, (img_m, img_n))
        )
        img_avg.set_cmap("gray")
        plt.axis("off")
        plt.title("Input face")

        ax2 = fig1.add_subplot(122)
        img_u1 = ax2.imshow(np.reshape(training_faces[:, recognized_faces[0][1]], (img_m, img_n)))
        img_u1.set_cmap("gray")
        plt.axis("off")
        plt.title(f"Match with {filenames[recognized_faces[0][1]][:7]}")

        plt.show()

    else:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        img_avg = ax1.imshow(np.reshape(test_face_norm + average_face, (img_m, img_n)))
        img_avg.set_cmap("gray")
        plt.axis("off")
        plt.title("Input face")

        ax2 = fig1.add_subplot(122)
        plt.axis("off")
        plt.title(f"Person is not in our data base")

        plt.show()


if __name__ == "__main__":
    main()
