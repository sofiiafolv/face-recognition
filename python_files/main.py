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
        help="0 - svd using power mothod, 1 - svd using library functions for EV's and EVc's, 2 - svd from numpy",
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

    # plt.imshow(np.reshape(average_face, (img_m, img_n)))

    # calculate SVD
    if svd_type == 0:
        U, sigma, VT = reduced_svd_using_qr(normalized_faces, k)
    elif svd_type == 1:
        U, sigma, VT = k_rank_approximation_from_scratch(normalized_faces, k)
    else:
        U, sigma, VT = np.linalg.svd(normalized_faces)
        U, sigma, VT = U[:, :k], sigma[:k], VT[:k, :]

    # project input image onto eigenfaces space
    # !!!!!!!! change to input path
    test_face_norm = training_faces[:, 0] - average_face
    projected_face_coord = U.T @ test_face_norm

    # calculate sigma @ VT to easily find coordinate vectors of training set images in the eigenfaces space
    sigma_VT = np.diag(sigma) @ VT

    flag = False
    # calculate distances between coordinate vectors of input face and training set images in the eigenfaces space
    for i in range(number_files):
        dist = np.linalg.norm(projected_face_coord - sigma_VT[:, i])
        if dist < 0:
            flag = True
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(121)
            img_avg = ax1.imshow(
                np.reshape(test_face_norm + average_face, (img_m, img_n))
            )
            img_avg.set_cmap("gray")
            plt.axis("off")
            plt.title("Input face")

            ax2 = fig1.add_subplot(122)
            img_u1 = ax2.imshow(np.reshape(training_faces[:, i], (img_m, img_n)))
            img_u1.set_cmap("gray")
            plt.axis("off")
            plt.title(f"Match with {filenames[i]}")

            plt.show()

    if not flag:
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