import numpy as np

from preprocessing_data import *
from svd import *
from svd_qr import *
from numpy import arange

k_list = [25, 50, 100, 200, 300]

def count_files(relative_path):
    current_directory = os.getcwd()
    directory_path = os.path.abspath(os.path.join(current_directory, relative_path))
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        file_count += len(files)

    return file_count


def running_with_threshold(threshold, U, sigma, VT):
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
            if dist < threshold:
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
    return (success/(success+failure), failure/(success+failure))

def finding_threshold(k, svd_type):
    training_faces, training_filenames = get_faces_matrix("../training_faces")
    average_face = np.mean(training_faces, axis=1)
    normalized_training_faces = (
            training_faces - np.tile(average_face, (training_faces.shape[1], 1)).T
    )
    if svd_type == 0:
        U, sigma, VT = reduced_svd_using_qr(normalized_training_faces, k)
    elif svd_type == 1:
        U, sigma, VT = k_rank_approximation_from_scratch(normalized_training_faces, k)
    else:
        U, sigma, VT = np.linalg.svd(normalized_training_faces, full_matrices=False)
        U, sigma, VT = U[:, :k], sigma[:k], VT[:k, :]
    effectiveness  = []
    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        print(i)
        effectiveness.append(running_with_threshold(i, U, sigma, VT))

    success_list = list(map(lambda x: x[0], effectiveness))
    failure_list = list(map(lambda x: x[1], effectiveness))

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    br1 = arange(len(success_list))
    br2 = [x + barWidth for x in br1]
    plt.title(f"Finding optimal threshold", fontdict={"fontsize": 30, "fontweight": 600})
    plt.bar(
        br1,
        success_list,
        color="palegreen",
        width=barWidth,
        label="success rate",
    )
    plt.bar(
        br2,
        failure_list,
        color="hotpink",
        width=barWidth,
        label="failure rate",
    )
    plt.xlabel("Threshold value", fontweight="bold", fontsize=15)
    plt.ylabel("Rate", fontweight="bold", fontsize=15)
    plt.xticks(
        [r + barWidth for r in range(len(failure_list))],
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    )
    plt.legend()
    plt.savefig(f"../graphs/thresholds.png")
    plt.show()

def main(k, svd_type):
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
        test_face = projected_face_coord[:, face] / np.linalg.norm(projected_face_coord[:, face])
        recognized_faces = []
        for i in range(number_files):
            dist = np.linalg.norm(test_face - sigma_VT[:, i] / np.linalg.norm(sigma_VT[:, i]))
            if dist < 0.5:
                recognized_faces.append((dist, i))

        if recognized_faces:
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
    return success / (success + failure)

def plot_effectiveness(effectiveness_svd_qr, effectiveness_svd_ev_ec, effectiveness_svd_numpy):
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    br1 = arange(len(effectiveness_svd_qr))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.title(f"Accuracy comparison", fontdict={"fontsize": 30, "fontweight": 600})
    plt.bar(
        br1,
        effectiveness_svd_qr,
        color="#F65058",
        width=barWidth,
        label="SVD using power method",
    )
    plt.bar(
        br2,
        effectiveness_svd_ev_ec,
        color="#FBDE44",
        width=barWidth,
        label="SVD using library functions for EV's and EVc's",
    )
    plt.bar(
        br3,
        effectiveness_svd_numpy,
        color="#26334A",
        width=barWidth,
        label="SVD from numpy",
    )
    plt.xlabel("k-rank", fontweight="bold", fontsize=15)
    plt.ylabel("Accuracy", fontweight="bold", fontsize=15)
    plt.xticks(
        [r + barWidth for r in range(len(effectiveness_svd_qr))],
        k_list,
    )
    plt.legend()
    plt.savefig(f"../graphs/effectiveness.png")
    plt.show()


if __name__ == "__main__":
    effectiveness_list = []
    for  i in [0,1,2]:
        e = []
        for j in k_list:
            print(i,j)
            e.append(main(j, i))
        effectiveness_list.append(e)

    plot_effectiveness(effectiveness_list[0], effectiveness_list[1], effectiveness_list[2])
