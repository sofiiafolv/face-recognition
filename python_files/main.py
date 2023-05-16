from preprocessing_data import *
from svd import *
from svd_qr import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="path to the photo to be recognized")
    parser.add_argument("k", type=int,
                        help="number k for best rank k approximation")
    parser.add_argument("svd", type=int, choices=[0, 1],
                        help="0 - svd using power mothod, 1 - svd using library functions for EV's and EVc's")
    args = parser.parse_args()
    return args.path, args.k, args.svd


def main():
    # parse arguments
    input_path, k, svd_type = parse_args()

    # form a matrix with all photos
    training_faces = get_faces_matrix('../training_faces')

    # find average face and deduct it from all photos
    average_face = np.mean(training_faces, axis=1)
    normalized_faces = training_faces - \
        np.tile(average_face, (training_faces.shape[1], 1)).T

    # plt.imshow(np.reshape(average_face, (img_m, img_n)))

    # calculate SVD
    if svd_type == 0:
        U, sigma, VT = reduced_svd_using_qr(normalized_faces, k)
    else:
        U, sigma, VT = k_rank_approximation_from_scratch(normalized_faces, k)

    # project input image onto eigenfaces space
    # !!!!!!!! change to input path
    test_face_norm = training_faces[:, 0] - average_face
    projected_face_coord = U.T @ test_face_norm

    # calculate sigma @ VT to easily find coordinate vectors of training set images in the eigenfaces space
    sigma_VT = np.diag(sigma) @ VT

    # calculate distances between coordinate vectors of input face and training set images in the eigenfaces space
    dist_list = []
    for i in range(640):
        dist = np.linalg.norm(projected_face_coord - sigma_VT[:, i])
        dist_list.append((dist, i // 64 + 1))

    print(sorted(dist_list))


if __name__ == "__main__":
    main()
