import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def get_disparity_map(rectified1_gray, rectified2_gray, min_disparity=-256, num_disparities=256, block_size=5, disp12_max_diff=-1):
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=disparity#StereoSGBM::StereoSGBM%28int%20minDisparity,%20int%20numDisparities,%20int%20SADWindowSize,%20int%20P1,%20int%20P2,%20int%20disp12MaxDiff,%20int%20preFilterCap,%20int%20uniquenessRatio,%20int%20speckleWindowSize,%20int%20speckleRange,%20bool%20fullDP%29
    # https://stackoverflow.com/questions/33688997/how-to-define-the-parameter-numdisparities-in-stereosgbm
    # https://github.com/guimeira/stereo-tuner
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, disp12MaxDiff=disp12_max_diff)
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, P1=8*3*blockSize*blockSize, p2=32*3*blockSize*blockSize)
    disparity = stereo.compute(rectified1_gray, rectified2_gray)
    disparity = (disparity/16).astype(np.int)
    return disparity


def plot_3d(pxyzs, colors):
    cand_idxs = np.random.choice(range(len(pxyzs)), size=int(len(pxyzs)/100))
    pxyzs = pxyzs[cand_idxs]
    colors = colors[cand_idxs]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pxyzs[:, 0], pxyzs[:, 1], pxyzs[:, 2], c=colors, cmap='gray', s=0.01)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



if __name__ =='__main__':
    import os 
    import cv2
    import numpy as np
    import TronGisPy as tgp
    import numpy.lib.recfunctions as nlr
    from matplotlib import pyplot as plt
    from io_aereo_params import get_DMC_aereo_params
    from TiePoints import find_tie_points_grids, plot_kp_lines, filter_tie_points_by_PQ_dist
    from Rectification import get_rectify_param, rectify, plot_rectified_img, plot_epipolar, rectify_idxs

    # get fp
    testdata_dir = os.path.join('Data', 'testcase3')
    img_fp1 = os.path.join(testdata_dir, '071021h_53_0042_refined.tif') # os.path.join(testdata_dir, "071021h_53~0042_hr4.tif")
    img_fp2 = os.path.join(testdata_dir, '071021h_53_0043_refined.tif') # os.path.join(testdata_dir, "071021h_53~0043_hr4.tif")
    aereo_params_fp1 = os.path.join(testdata_dir, '071021h_53_0042_refined.pkl')
    aereo_params_fp2 = os.path.join(testdata_dir, '071021h_53_0043_refined.pkl')

    # read data
    ras1 = tgp.read_raster(img_fp1)
    ras2 = tgp.read_raster(img_fp2)
    aereo_params1 = get_DMC_aereo_params(aereo_params_fp1, ras1.shape)
    aereo_params2 = get_DMC_aereo_params(aereo_params_fp2, ras2.shape)
    
    # preprocessing
    img1_norm = tgp.Normalizer().fit_transform(ras1.data[:, :, :3], clip_percentage=(0.1, 0.9))
    img2_norm = tgp.Normalizer().fit_transform(ras2.data[:, :, :3], clip_percentage=(0.1, 0.9))
    img1 = (tgp.Normalizer().fit_transform(ras1.data[:, :, [2, 1, 0]]) * 255).astype(np.uint8)
    img2 = (tgp.Normalizer().fit_transform(ras2.data[:, :, [2, 1, 0]]) * 255).astype(np.uint8)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find tie points & filter tie points
    kp1_pts, kp2_pts = find_tie_points_grids(gray1, gray2, nfeatures=1000, topn_n_matches=700, grids=(3, 3)) # find_tie_points_farest(gray1, gray2)
    dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=5)

    # rectify image
    mapx1, mapy1, mapx2, mapy2 = get_rectify_param(gray1.shape, kp1_pts, kp2_pts, shearing=True)
    rectified1 = rectify(img1, mapx1, mapy1, border_value=0)
    rectified2 = rectify(img2, mapx2, mapy2, border_value=0)
    rectified1_norm = rectify(img1_norm, mapx1, mapy1, border_value=0)
    rectified2_norm = rectify(img2_norm, mapx2, mapy2, border_value=0)

    # rectify image index
    rectified_npidxs1 = rectify_idxs(img1, mapx1, mapy1, border_value=-1)
    rectified_npidxs2 = rectify_idxs(img2, mapx2, mapy2, border_value=-1)

    # kp1_idx_tuple = nlr.unstructured_to_structured(kp1_pts.astype(np.int)).astype('O')
    # rec_idx_tuple = nlr.unstructured_to_structured(rectified_npidxs1.reshape(-1, 2)).astype('O')
    # cand_npidxs = set(kp1_idx_tuple) & set(rec_idx_tuple)


    # disparity_map
    # disparity = get_disparity_map(rectified1, rectified2)
    # plt.hist(disparity.flatten())
    # plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(10, 20))
    for i, min_disparity in enumerate([-64, -128, -256, -384, -512]):
        disparity = get_disparity_map(rectified1, rectified2, min_disparity=min_disparity)
        axes[0][i].imshow(disparity, cmap='gray')
        axes[0][i].set_title("md" + str(min_disparity))
        axes[0][i].axis('off')
    for i, num_disparities in enumerate([64, 128, 256, 384, 512]):
        disparity = get_disparity_map(rectified1, rectified2, num_disparities=num_disparities)
        axes[1][i].imshow(disparity, cmap='gray')
        axes[1][i].set_title("nd" + str(num_disparities))
        axes[1][i].axis('off')
    plt.show()

    # # fig, axes = plt.subplots(3, 5, figsize=(10, 100))
    # # for i, num_disparities in enumerate([32, 64, 128]):
    # #     for j, block_size in enumerate([5, 9, 13, 15, 19]):
    # #         disparity = get_disparity_map(rectified1, rectified2, num_disparities=num_disparities, block_size=block_size)
    # #         axes[i, j].imshow(disparity, cmap='gray')
    # #         axes[i, j].set_title("b" + str(block_size) + "_d" + str(num_disparities))
    # #         axes[i, j].axis('off')
    # # plt.show()

    # # axes[-1].imshow(cv2.addWeighted(rectified1_norm, 0.5, rectified2_norm, 0.5, 0))

