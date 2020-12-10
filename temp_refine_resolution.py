import os
import cv2
import pickle 
import numpy as np
import TronGisPy as tgp
from io_aereo_params import get_DMC_aereo_params

img_shape = (3456, 1920)
raster_fp1 = os.path.join('Data', 'testcase2','071021h_53_0042_refined.tif')
aereo_params_fp1 = os.path.join('Data', 'testcase2', '071021h_53_0042_refined.pkl')
raster_fp2 = os.path.join('Data', 'testcase2','071021h_53_0043_refined.tif')
aereo_params_fp2 = os.path.join('Data', 'testcase2', '071021h_53_0043_refined.pkl')

raster1 = tgp.read_raster(raster_fp1)
aerotri_params1 = np.array(get_DMC_aereo_params(aereo_params_fp1, img_shape))
raster2 = tgp.read_raster(raster_fp2)
aerotri_params2 = np.array(get_DMC_aereo_params(aereo_params_fp2, img_shape))

# resize
rows, cols = raster1.shape[:2]
raster1.data = cv2.resize(raster1.data, (int(cols/2), int(rows/2)))
rows, cols = raster2.shape[:2]
raster2.data = cv2.resize(raster2.data, (int(cols/2), int(rows/2)))

# [OPK1, L_XYZ1, DMC_ROWS_LABEL, DMC_COLS_LABEL, DMC_FOCAL_LENGTH, DMC_PIXEL_SIZE]
aerotri_params1[2] /= 2 # DMC_ROWS_LABEL
aerotri_params1[3] /= 2 # DMC_COLS_LABEL
aerotri_params1[4] *= 2 # DMC_PIXEL_SIZE
aerotri_params2[2] /= 2 # DMC_ROWS_LABEL
aerotri_params2[3] /= 2 # DMC_COLS_LABEL
aerotri_params2[4] *= 2 # DMC_PIXEL_SIZE


raster1.to_file(os.path.join('Data', 'testcase3', '071021h_53_0042_refined.tif'))
raster2.to_file(os.path.join('Data', 'testcase3', '071021h_53_0043_refined.tif'))
with open(os.path.join('Data', 'testcase3', '071021h_53_0042_refined.pkl'), 'wb') as f:
    pickle.dump(tuple(aerotri_params1), f)
with open(os.path.join('Data', 'testcase3', '071021h_53_0043_refined.pkl'), 'wb') as f:
    pickle.dump(tuple(aerotri_params2), f)

