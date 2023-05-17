import numpy as np

from preprocessing_data import *
from svd import *
from svd_qr import *
import argparse




def count_files(relative_path):
    current_directory = os.getcwd()
    directory_path = os.path.abspath(os.path.join(current_directory, relative_path))
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)

    return file_count


def main(k, svd_type):
    # parse arguments

    # form a matrix with all photos: for training and testing
    training_faces, training_filenames = get_faces_matrix("../training_faces")
    number_files = count_files("../training_faces")
    print(count_files)

    testing_faces, testing_filenames = get_faces_matrix("../testing_faces")

    # find average face and deduct it from all photos
    average_face = np.mean(training_faces, axis=1)
    normalized_training_faces = (
        training_faces - np.tile(average_face, (training_faces.shape[1], 1)).T
    )
    normalized_testing_faces = (
        testing_faces - np.tile(average_face, (testing_faces.shape[1], 1)).T
    )

    # calculate SVD
    if svd_type == 0:
        U, sigma, VT = reduced_svd_using_qr(normalized_training_faces, k)
    elif svd_type == 1:
        U, sigma, VT = k_rank_approximation_from_scratch(normalized_training_faces, k)
    else:
        U, sigma, VT = np.linalg.svd(normalized_training_faces, full_matrices=False)
        U, sigma, VT = U[:, :k], sigma[:k], VT[:k, :]

    # project input image onto eigenfaces space
    projected_face_coord = U.T @ normalized_testing_faces

    # calculate sigma @ VT to easily find coordinate vectors of training set images in the eigenfaces space
    sigma_VT = np.diag(sigma) @ VT

    # calculate distances between coordinate vectors of input face and training set images in the eigenfaces space
    success = 0
    failure = 0
    for face in range(projected_face_coord.shape[1]):
        test_face = projected_face_coord[:,face]/np.linalg.norm(projected_face_coord[:, face])
        recognized_faces = []
        for i in range(number_files):
            dist = np.linalg.norm(test_face - sigma_VT[:, i]/np.linalg.norm(sigma_VT[:, i]))
            if dist < 0.5:
                recognized_faces.append((dist, i))

        if  recognized_faces:
            recognized_faces.sort()

            if testing_filenames[face][:7] == training_filenames[recognized_faces[0][1]][:7]:
                success += 1
            else:
                failure += 1
        else:
            if int(testing_filenames[face][5:7]) in range(1, 17):
                failure += 1
            else:
                success += 1
    return success/(success+failure)




if __name__ == "__main__":
    main(200, 0)
