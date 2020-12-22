import pickle

def get_DMC_aereo_params(pke_fp, img_shape=(13824, 7680)):
    with open(pke_fp, 'rb')as f:
        OPK, L_XYZ, _, _, _, _ = pickle.load(f)
    DMC_ROWS_LABEL, DMC_COLS_LABEL = img_shape[:2]
    DMC_PIXEL_SIZE = 92.16/DMC_COLS_LABEL # pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    DMC_FOCAL_LENGTH = 120 # 0.012 mm / pixel
    XOFFSET, YOFFSET = 0, 0
    return OPK, L_XYZ, DMC_ROWS_LABEL, DMC_COLS_LABEL, DMC_FOCAL_LENGTH, DMC_PIXEL_SIZE, XOFFSET, YOFFSET

if __name__ == '__main__':
    import os
    img_shape = (3456, 1920)
    aereo_params_fp = os.path.join('Data', 'testcase3', '071021h_53_0042_refined.pkl')
    aereo_params = get_DMC_aereo_params(aereo_params_fp, img_shape)
    print(aereo_params)
    img_shape = (3456, 1920)
    aereo_params_fp = os.path.join('Data', 'testcase3', '071021h_53_0043_refined.pkl')
    aereo_params = get_DMC_aereo_params(aereo_params_fp, img_shape)
    print(aereo_params)


