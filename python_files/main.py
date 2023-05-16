from preprocessing_data import *
from svd import *
from svd_qr import *


def main():
    # form a matrix with all photos
    training_faces = get_faces_matrix('./training_faces')

    # find average face and deduct it from all photos
    average_face = np.mean(training_faces,axis=1)
    normalized_faces = training_faces - np.tile(average_face,(training_faces.shape[1],1)).T

    plt.imshow(np.reshape(average_face,(img_m, img_n)))


    U, sigma, VT = reduced_svd(normalized_faces)
    sigma_VT = np.diag(sigma) @ VT

    test_face = training_faces[:,0] - average_face
    projected_face_coord = U.T @ test_face

    for i in range(64, 640):
        print(np.linalg.norm(projected_face_coord - sigma_VT[:, i]))

    U_qr, sigma_qr, VT_qr = reduced_svd_using_qr(normalized_faces)
    sigma_VT_qr = np.diag(sigma_qr) @ VT_qr

    test_face = training_faces[:,0] - average_face
    projected_face_coord = U_qr.T @ test_face

    results = []
    for i in range(640):
        dist = np.linalg.norm(projected_face_coord - sigma_VT_qr[:, i])
        # print(dist)
        results.append((dist, i // 64 + 1))
        
    print(sorted(results))

    


if __name__ == "__main__":
    main()
