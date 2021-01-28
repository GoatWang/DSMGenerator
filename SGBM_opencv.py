import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
try:
    from DSMGenerator.TiePoints import get_dist_from_tie_points, find_tie_points_stereo_grids, find_tie_points_grids_matching
    from DSMGenerator.Rectification import get_rectify_param, rectify, plot_rectified_img, plot_epipolar, rectify_idxs, rectify_kps, unrectify_kps
except:
    from TiePoints import get_dist_from_tie_points, find_tie_points_stereo_grids, find_tie_points_grids_matching
    from Rectification import get_rectify_param, rectify, plot_rectified_img, plot_epipolar, rectify_idxs, rectify_kps, unrectify_kps

def get_disparity_map(rectified1_gray, rectified2_gray, min_disparity, num_disparities, block_size):
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=disparity#StereoSGBM::StereoSGBM%28int%20minDisparity,%20int%20numDisparities,%20int%20SADWindowSize,%20int%20P1,%20int%20P2,%20int%20disp12MaxDiff,%20int%20preFilterCap,%20int%20uniquenessRatio,%20int%20speckleWindowSize,%20int%20speckleRange,%20bool%20fullDP%29
    # https://stackoverflow.com/questions/33688997/how-to-define-the-parameter-numdisparities-in-stereosgbm
    # https://github.com/guimeira/stereo-tuner
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size)
    win_size = 5
    stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, #disp12MaxDiff=disp12_max_diff,
                                    uniquenessRatio = 5,
                                    speckleWindowSize = 50,
                                    speckleRange = 1,
                                    P1 = 8*3*win_size**2,
                                    P2 =32*3*win_size**2)
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, P1=8*3*blockSize*blockSize, p2=32*3*blockSize*blockSize)
    disparity = stereo.compute(rectified1_gray, rectified2_gray)
    disparity = (disparity/16)
    return disparity

# def get_disparity_map_grid(rectified1_gray, rectified2_gray, x_st, x_end, , min_disparity, num_disparities, block_size):
#     win_size = 5
#     stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, disp12MaxDiff=disp12_max_diff,
#                                     uniquenessRatio = 5,
#                                     speckleWindowSize = 50,
#                                     speckleRange = 1,
#                                     P1 = 8*3*win_size**2,
#                                     P2 =32*3*win_size**2)
#     # stereo = cv2.StereoSGBM_create(minDisparity=min_disparity, numDisparities=num_disparities, blockSize=block_size, P1=8*3*blockSize*blockSize, p2=32*3*blockSize*blockSize)
#     disparity = stereo.compute(rectified1_gray, rectified2_gray)
#     disparity = (disparity/16)
#     return disparity

def plot_disparity_verification(disparity, min_disparity, kp1_pts, kp2_pts):
    kp1_pts_int, kp2_pts_int = kp1_pts.astype(np.int), kp2_pts.astype(np.int)
    valid_disparity_mask = (disparity[kp1_pts_int[:, 1], kp1_pts_int[:, 0]]) != (min_disparity-1)
    kp1_pts_int, kp2_pts_int = kp1_pts_int[valid_disparity_mask], kp2_pts_int[valid_disparity_mask]

    disparity_sgm = disparity[kp1_pts_int[:, 1], kp1_pts_int[:, 0]]
    disparity_sift = kp1_pts_int[:, 0] - kp2_pts_int[:, 0]
    disparity_diff = disparity_sift - disparity_sgm
    disparity_thres = np.quantile(np.abs(disparity_diff), 0.8)

    fig, ax = plt.subplots(1, 1, figsize=(5, 15))
    im = ax.imshow(disparity, cmap='gray')
    gen_color = lambda x: (1, 0, 0) if np.abs(x)>disparity_thres else (0, 1, 0)
    colors = [gen_color(d) for d in disparity_diff]
    ax.scatter(kp1_pts_int[:, 0], kp1_pts_int[:, 1], c=colors, s=3)

    dist_strs = np.array([str(int(ds)) + ", " + "%.2f"%dd  for ds, dd in zip(disparity_sift, disparity_diff)])
    rand_idxs = np.random.choice(range(len(kp1_pts_int)), 400)
    for idx, (kp1_pt, dist_str) in enumerate(zip(kp1_pts_int, dist_strs)):
        if idx in rand_idxs:
            ax.text(kp1_pt[0]-3, kp1_pt[1], dist_str, color=colors[idx])
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("Disparity")
    plt.show()

def generate_cloud_point(disparity, min_disparity, border_value, rectified1, rectified2, img1, img2, H1, H2, aereo_params1, aereo_params2, padding=0, valid_w_tiepoints=False):
    """
    padding: pixels to Indentation
    """
    assert img1.dtype == np.uint8, "img1 be in np.uint8 type"
    assert img2.dtype == np.uint8, "img2 should be in np.uint8 type"
    assert padding >= 0, "padding should be greater than or equal  zero"
    ## generate idxs
    npidxs_x1, npidxs_y1 = np.meshgrid(range(img1.shape[1]), range(img1.shape[0])) # (3000, 4096)
    npidxs_x2, npidxs_y2 = np.meshgrid(range(img2.shape[1]), range(img2.shape[0])) # (3000, 4096)
    npidxs_x2 = npidxs_x2 - disparity
    npidxs_x1, npidxs_y1 = npidxs_x1.ravel(), npidxs_y1.ravel() # (3000, 4096)
    npidxs_x2, npidxs_y2 = npidxs_x2.ravel(), npidxs_y2.ravel() # (3000, 4096)
    colors = img1.reshape(-1, 3) if img1.shape[-1] == 3 else np.stack([img1.ravel()]*3).T

    # out_bound_mask (delete when out_bound the buffer after adding disparity)
    x_st, x_end, y_st, y_end = padding, img1.shape[1]-padding, padding, img1.shape[0]-padding
    in_bound_mask = (npidxs_x1 >= x_st) & (npidxs_x1 < x_end) & (npidxs_y1 >= y_st) & (npidxs_y1 < y_end) # padding
    in_bound_mask &= (npidxs_x2 >= 0) & (npidxs_x2 < img1.shape[1]) # out bounds when convert to img2
    in_bound_mask &= disparity.ravel() != (min_disparity-1) # out_bound_mask_nan
    in_bound_mask &= rectified1[:, :, 0].ravel() != border_value # out_bound_mask_nan
    # in_bound_mask &= rectified2.ravel() != border_value # out_bound_mask_nan
    npidxs_x1, npidxs_y1, npidxs_x2, npidxs_y2, colors = npidxs_x1[in_bound_mask], npidxs_y1[in_bound_mask], npidxs_x2[in_bound_mask], npidxs_y2[in_bound_mask], colors[in_bound_mask]

    # get unrectified kps
    kp1 = unrectify_kps(np.stack([npidxs_x1, npidxs_y1]).T, H1)
    kp2 = unrectify_kps(np.stack([npidxs_x2, npidxs_y2]).T, H2)

    # valid disparity with sift on non-recitified image
    if valid_w_tiepoints:
        # sift
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1_pts, kp2_pts = find_tie_points_grids_matching(gray1, gray2, nfeatures=4000, topn_n_matches=300, grids=(3, 3)) # find_tie_points_farest(gray1, gray2)
        _, _, _, _, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=5)
        kp1_pts, kp2_pts = kp1_pts.astype(np.int), kp2_pts.astype(np.int)
        # plot_kp_lines(gray1, kp1_pts, gray2, kp2_pts)

        # sgm
        disparity_grid = np.zeros_like(gray1, dtype=np.int16)
        kp1_int = kp1.astype(np.int)
        kp1_int[kp1_int[:, 0]>=gray1.shape[1]] = gray1.shape[1] - 1
        kp1_int[kp1_int[:, 1]>=gray1.shape[0]] = gray1.shape[0] - 1
        disparity_grid[kp1_int[:, 1], kp1_int[:, 0]] = np.sqrt(np.sum(np.square(kp2 - kp1), axis=1)).astype(np.int)

        # difference
        sgm_disparity = disparity_grid[kp1_pts[:, 1], kp1_pts[:, 0]]
        sift_disparity = np.sqrt(np.sum(np.square(kp2_pts - kp1_pts), axis=1))
        disparity_diff = sgm_disparity - sift_disparity
        plt.hist(disparity_diff)
        plt.show()

        # showing
        disparity_grid_showing = disparity_grid.copy()
        print('min disparity', np.min(disparity_grid_showing[disparity_grid_showing!=0]))
        disparity_grid_showing[disparity_grid_showing!=0] -= np.min(disparity_grid_showing[disparity_grid_showing!=0])
        fig, ax = plt.subplots(1, 1)
        ax.imshow(disparity_grid_showing, cmap='gray')
        gen_color = lambda x: (1, 0, 0) if np.abs(x)>50 else (0, 1, 0)
        ax.scatter(kp1_pts[:, 0], kp1_pts[:, 1], c=[gen_color(d) for d in disparity_diff], s=3)
        dist_strs = np.array([str(int(d)) for d in np.sqrt(np.sum(np.square(kp2_pts - kp1_pts), axis=1))])
        rand_idxs = np.random.choice(range(len(kp1_pts)), 300)
        for idx, (kp1_pt, dist_str) in enumerate(zip(kp1_pts, dist_strs)):
            if idx in rand_idxs:
                ax.text(kp1_pt[0]-100, kp1_pt[1], dist_str)
        plt.show()

    # without pixel value
    dep1, dep2, pxyz1, pxyz2, dists_ecu = get_dist_from_tie_points(aereo_params1, aereo_params2, kp1, kp2)
    # dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1, kp2, max_dist=0.02)
    return pxyz1, pxyz2, colors, dists_ecu

# using rectified indeixes
# def generate_cloud_point(disparity, nan_value, rectified1, rectified2, img1, img2, mapx1, mapx2, mapy1, mapy2, aereo_params1, aereo_params2, padding=0, valid_w_tiepoints=False):
#     """
#     padding: pixels to Indentation
#     """
#     assert img1.dtype == np.uint8, "img1 should be in np.uint8 type"
#     assert img2.dtype == np.uint8, "img2 should be in np.uint8 type"
#     assert padding >= 0, "padding should be greater than or equal  zero"
#     ## generate idxs
#     npidxs_x1, npidxs_y1 = np.meshgrid(range(rectified1.shape[1]), range(rectified1.shape[0])) # (3000, 4096)
#     npidxs_x2, npidxs_y2 = np.meshgrid(range(rectified2.shape[1]), range(rectified2.shape[0])) # (3000, 4096)
#     npidxs_x2 = npidxs_x2 + disparity
#     npidxs_x1, npidxs_y1 = npidxs_x1.ravel(), npidxs_y1.ravel() # (3000, 4096)
#     npidxs_x2, npidxs_y2 = npidxs_x2.ravel(), npidxs_y2.ravel() # (3000, 4096)
#     colors = img1.reshape(-1, 3) if img1.shape[-1] == 3 else np.stack([img1.ravel()]*3).T  # TODO: change to 3 bands, 12288000

#     # out_bound_mask
#     x_st, x_end, y_st, y_end = padding, img1.shape[1]-padding, padding, img1.shape[0]-padding
#     out_bound_mask = (npidxs_x1 > x_st) & (npidxs_x1 < x_end) & (npidxs_y1 > y_st) & (npidxs_y1 < y_end) # padding
#     out_bound_mask &= (npidxs_x2 >= 0) & (npidxs_x2 < rectified1.shape[1]) # out bounds when convert to img2
#     out_bound_mask &= disparity.ravel() != nan_value # out_bound_mask_nan
#     npidxs_x1, npidxs_y1, npidxs_x2, npidxs_y2, colors = npidxs_x1[out_bound_mask], npidxs_y1[out_bound_mask], npidxs_x2[out_bound_mask], npidxs_y2[out_bound_mask], colors[out_bound_mask]

#     # rectify image index
#     rectified_xys1 = rectify_idxs(img1, mapx1, mapy1, border_value=-1)
#     rectified_xys2 = rectify_idxs(img2, mapx2, mapy2, border_value=-1)
#     kp1 = rectified_xys1[npidxs_y1, npidxs_x1]
#     kp2 = rectified_xys2[npidxs_y2, npidxs_x2]

#     # valid disparity with sift on non-recitified image
#     if valid_w_tiepoints:
#         gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#         kp1_pts, kp2_pts = find_tie_points_stereo_grids(gray1, gray2, nfeatures=1000, topn_n_matches=100, grids=(1, 1))
#         kp1_pts, kp2_pts = kp1_pts.astype(np.int), kp2_pts.astype(np.int)

#         disparity_grid = np.zeros_like(gray1, dtype=np.int16)
#         kp1_int = kp1.astype(np.int)
#         kp1_int[kp1_int[:, 0]>=gray1.shape[1]] = gray1.shape[1] - 1
#         kp1_int[kp1_int[:, 1]>=gray1.shape[0]] = gray1.shape[0] - 1
#         disparity_grid[kp1_int[:, 1], kp1_int[:, 0]] = np.sqrt(np.sum(np.square(kp2 - kp1), axis=1)).astype(np.int)
#         # disparity_grid[disparity_grid>300] = 0

#         sgm_disparity = disparity_grid[kp1_pts[:, 1], kp1_pts[:, 0]]
#         sift_disparity = np.sqrt(np.sum(np.square(kp2_pts - kp1_pts), axis=1))
#         disparity_diff = sgm_disparity - sift_disparity
#         plt.hist(disparity_diff)
#         plt.show()

#         fig, ax = plt.subplots(1, 1)
#         ax.imshow(disparity_grid, cmap='gray')
#         gen_color = lambda x: (1, 0, 0) if x>50 else (0, 1, 0)
#         ax.scatter(kp1_pts[:, 0], kp1_pts[:, 1], c=[gen_color(d) for d in disparity_diff])
#         dist_strs = np.array([str(int(d)) for d in np.sqrt(np.sum(np.square(kp2_pts - kp1_pts), axis=1))])
#         for idx, (kp1_pt, dist_str) in enumerate(zip(kp1_pts, dist_strs)):
#             ax.text(kp1_pt[0]-100, kp1_pt[1], dist_str)
#         plt.show()

#     # without pixel value
#     out_bound_mask = (kp1[:, 0] != -1) & (kp1[:, 1] != -1) & (kp2[:, 0] != -1) & (kp2[:, 1] != -1) # black area of original image
#     kp1, kp2, colors = kp1[out_bound_mask], kp2[out_bound_mask], colors[out_bound_mask]
#     dep1, dep2, pxyz1, pxyz2, dists_ecu = get_dist_from_tie_points(aereo_params1, aereo_params2, kp1, kp2)
#     # dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1, kp2, max_dist=0.02)
#     return pxyz1, pxyz2, colors, dists_ecu

def plot_3d(pxyzs, colors):
    assert colors.shape[-1] == 3, "colors should be in (None, 3) dimension"
    assert colors.dtype == np.uint8, "colors should be in np.uint8 type"

    cand_idxs = np.random.choice(range(len(pxyzs)), size=int(len(pxyzs)/100))
    pxyzs = pxyzs[cand_idxs]
    colors = colors[cand_idxs] / 255

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pxyzs[:, 0], pxyzs[:, 1], pxyzs[:, 2], c=colors, cmap='gray', s=0.01)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def write_las(pxyzs, colors, aereo_params_1, aereo_params_2, las_fp, convert_to_potree=False):
    import TronLasPy as lp
    assert colors.shape[-1] == 3, "colors should be in (None, 3) dimension"
    assert colors.dtype == np.uint8, "colors should be in np.uint8 type"

    # write cloud point
    cp_data = np.empty((len(pxyzs)+2, 10)) # x, y, z, num_returns, return_num, intensity, classification, red, green, blue
    cp_data[:, [3, 4, 6]] = 1
    cp_data[:len(pxyzs), :3] = pxyzs
    cp_data[:len(pxyzs), 5] = np.mean(colors, axis=1)
    cp_data[:len(pxyzs), [7, 8, 9]] = colors

    # write L_XYZ
    f1, f2 = aereo_params_1[1], aereo_params_2[1] #OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE, XOFFSET, YOFFSET = aerotri_params
    cp_data[[-2, -1], :3] = f1, f2
    cp_data[[-2, -1], 5] = 0, 0
    cp_data[[-2, -1], -3:] = np.array([[255, 0, 0], [0, 0, 255]])

    # write las
    cp = lp.CloudPoint(cp_data, offset=None, scale=(0.0001, 0.0001, 0.0001))
    cp.to_file(las_fp)

    if convert_to_potree:
        lp.convert_to_potree(las_fp, classification=True, overwrite=True)

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
    img1 = (tgp.Normalizer().fit_transform_opencv(ras1.data[:, :, :3], clip_percentage=(0.1, 0.9), out_dtype=np.float64) * 255).astype(np.uint8)
    img2 = (tgp.Normalizer().fit_transform_opencv(ras2.data[:, :, :3], clip_percentage=(0.1, 0.9), out_dtype=np.float64) * 255).astype(np.uint8)
    # img1 = (tgp.Normalizer().fit_transform(ras1.data[:, :, [2, 1, 0]]) * 255).astype(np.uint8)
    # img2 = (tgp.Normalizer().fit_transform(ras2.data[:, :, [2, 1, 0]]) * 255).astype(np.uint8)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find tie points & filter tie points
    kp1_pts, kp2_pts = find_tie_points_grids_matching(gray1, gray2, nfeatures=4000, topn_n_matches=300, grids=(3, 3)) # find_tie_points_farest(gray1, gray2)
    dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=5)
    plot_kp_lines(img1, kp1_pts, img2, kp2_pts, title="TiePoints", figsize=(10, 10))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.hist(kp1_pts[:, 0] - kp2_pts[:, 0])
    # ax1.set_title('x')
    # ax2.hist(kp1_pts[:, 1] - kp2_pts[:, 1])
    # ax2.set_title('y')
    # plt.show()

    # rectify image
    border_value = 0
    H1, H2, mapx1, mapy1, mapx2, mapy2 = get_rectify_param(gray1.shape, kp1_pts, kp2_pts, shearing=True)
    rectified1 = rectify(img1, mapx1, mapy1, border_value=border_value)
    rectified2 = rectify(img2, mapx2, mapy2, border_value=border_value)
    plot_rectified_img(rectified1, rectified2, line_interval=30)

    kp1_pts_rec = rectify_kps(kp1_pts, H1)
    kp2_pts_rec = rectify_kps(kp2_pts, H2)
    x_diff_min, x_diff_max = np.min(kp1_pts_rec[:, 0] - kp2_pts_rec[:, 0]), np.max(kp1_pts_rec[:, 0] - kp2_pts_rec[:, 0])
    min_disparity = int(np.floor(x_diff_min))
    num_disparities = int(np.ceil(x_diff_max - x_diff_min))
    num_disparities = num_disparities + (16 - (num_disparities % 16))
    nan_value = min_disparity - 1
    plot_kp_lines(rectified1, kp1_pts_rec, rectified2, kp2_pts_rec)
    # plot_kp_lines(rectified1, kp1_pts_rec, rectified2, kp2_pts_rec, title="TiePoints", figsize=(10, 10), texts_left=['%.2f'%x for x in (kp1_pts_rec[:, 0] - kp2_pts_rec[:, 0])])
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.hist(kp2_pts_rec[:, 0] - kp1_pts_rec[:, 0])
    # ax1.set_title('x')
    # ax2.hist(kp2_pts_rec[:, 1] - kp1_pts_rec[:, 1])
    # ax2.set_title('y')
    # plt.show()

    # disparity_map
    disparity = get_disparity_map(rectified1, rectified2, min_disparity=min_disparity, num_disparities=num_disparities, block_size=5)
    plt.imshow(disparity, cmap='gray')
    plt.show()
    # plot_disparity_verification(disparity, min_disparity, kp1_pts_rec, kp2_pts_rec)

    # generate cloud points 
    pxyz1, pxyz2, colors, dists_ecu = generate_cloud_point(disparity, min_disparity, border_value, rectified1, rectified2, img1, img2, H1, H2, aereo_params1, aereo_params2, padding=30, valid_w_tiepoints=False)
    plot_3d(pxyz1, colors)

    # convert_to_potree = True
    # las_fp = os.path.join('temp', 'airphoto_test.las')
    # write_las(pxyz1, colors, aereo_params1, aereo_params2, las_fp, convert_to_potree=convert_to_potree)

