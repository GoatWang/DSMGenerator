import os 
import cv2
import numpy as np
import TronGisPy as tgp
from matplotlib import pyplot as plt
from io_aereo_params import get_DMC_aereo_params
from TiePoints import find_tie_points_grids, plot_kp_lines, filter_tie_points_by_PQ_dist
from Rectification import get_rectify_param, rectify, plot_rectified_img, plot_epipolar, rectify_idxs

# get fp
# testdata_dir = os.path.join('Data', 'testcase2')
# img_fp1 = os.path.join(testdata_dir, '071021h_53_0042' + "_refined.tif") # os.path.join(testdata_dir, "071021h_53~0042_hr4.tif")
# img_fp2 = os.path.join(testdata_dir, '071021h_53_0043' + "_refined.tif") # os.path.join(testdata_dir, "071021h_53~0043_hr4.tif")
# aereo_params_fp1 = os.path.join('Data', 'testcase2', '071021h_53_0042_refined.pkl')
# aereo_params_fp2 = os.path.join('Data', 'testcase2', '071021h_53_0043_refined.pkl')
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

# rectify image & image index
mapx1, mapy1, mapx2, mapy2 = get_rectify_param(gray1.shape, kp1_pts, kp2_pts, shearing=True)
rectified1_gray = rectify(gray1, mapx1, mapy1, border_value=0)
rectified2_gray = rectify(gray2, mapx2, mapy2, border_value=0)
rectified1_norm = rectify(img1_norm, mapx1, mapy1, border_value=0)
rectified2_norm = rectify(img2_norm, mapx2, mapy2, border_value=0)
rectified_npidxs1 = rectify_idxs(img1, mapx1, mapy1, border_value=0)
rectified_npidxs2 = rectify_idxs(img2, mapx2, mapy2, border_value=0)

# # SGBM
# stereo = cv2.StereoSGBM_create(minDisparity=-64, numDisparities=128, blockSize=5)
# disparity = stereo.compute(rectified1_gray, rectified2_gray)
# fig, axes = plt.subplots(1, 3, figsize=(15, 15))
# axes[0].imshow(rectified1_norm)
# axes[1].imshow(rectified2_norm)
# axes[2].imshow(disparity, cmap='gray')
# plt.show()





# SGM find disparity map
# https://github.com/2b-t/SciPy-stereo-sgm/blob/master/Main.ipynb


# L_XYZ1, L_XYZ2 = aereo_params1[1], aereo_params2[1]
# dist = np.sqrt(np.sum(np.square(np.array(L_XYZ1) - np.array(L_XYZ2))))
# stereo = cv2.StereoSGBM_create()
# disparity = stereo.compute(rectified1, rectified2).astype(np.float32)
# print(disparity.shape)
# print(np.min(disparity), np.max(disparity))
# plt.imshow(disparity, cmap='gray')
# plt.show()

