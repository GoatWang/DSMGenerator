import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
try:
    from DSMGenerator.AeroTriangulation import get_PQ, get_line_vecs, project_npidxs_to_XYZs_by_k
except:
    from AeroTriangulation import get_PQ, get_line_vecs, project_npidxs_to_XYZs_by_k

def find_tie_points_grids(gray1, gray2, nfeatures=1000, topn_n_matches=300, grids=(1, 1)):
    """split the image into grids to find the descriptors to ensure the well distributed tie points.
    
    Parameters
    ----------
    gray1: ndarray
        The first image with only one channel.
    gray2: ndarray
        The second image with only one channel.
    nfeatures: int
        The number of descriptor to find in one image.
    topn_n_matches: int, optional, default: 300
        The top n match links to get the tie point pairs.
    grids: tuple of int, optional, default: (1, 1)
        The girdspec to split the image. The Length of grids should be 2, which 
        means (split_rows, split_cols).

    Returns
    -------
    kp1s_pts: ndarray
        The shape of kp1s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2s_pts: ndarray
        The shape of kp2s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    """
    h, w = gray1.shape
    grid_h, grid_w = int(h//grids[0]), int(w//grids[1])
    kp1s, kp2s, des1s, des2s = [], [], [], []
    for grid_row_idx in range(grids[0]):
        for grid_col_idx in range(grids[1]):
            # orb = cv2.xfeatures2d.SURF_create(nfeatures)
            orb = cv2.ORB_create(nfeatures)
            # orb = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers=10)
            row_st, row_end = grid_row_idx * grid_h, (grid_row_idx+1) * grid_h
            col_st, col_end = grid_col_idx * grid_w, (grid_col_idx+1) * grid_w
            mask = np.zeros_like(gray1)
            mask[row_st:row_end, col_st:col_end] = 1
            kp1, des1 = orb.detectAndCompute(gray1, mask)
            kp2, des2 = orb.detectAndCompute(gray2, mask)
            if des1 is not None:
                kp1s.extend(kp1); des1s.extend(des1);  
            if des2 is not None:
                kp2s.extend(kp2); des2s.extend(des2)
    des1s = np.stack(des1s).astype(np.uint8)
    des2s = np.stack(des2s).astype(np.uint8)

    # match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1s, des2s) # Match descriptors.
    matches = sorted(matches, key=lambda x:x.distance)

    # convert Matches and kps into numpy array and list
    kp1s_pts = np.array([kp.pt for kp in kp1s])
    kp2s_pts = np.array([kp.pt for kp in kp2s])
    kp1_paired_idxs = [m.queryIdx for m in matches[:topn_n_matches]]
    kp2_paired_idxs = [m.trainIdx for m in matches[:topn_n_matches]]
    kp1s_pts = np.array(kp1s_pts)[kp1_paired_idxs]
    kp2s_pts = np.array(kp2s_pts)[kp2_paired_idxs]
    return kp1s_pts, kp2s_pts

def generate_grid_idxs(distparity, buffer, grid_idx, interval):
    left_first = distparity <= 0 # or top first
    distparity = -np.abs(distparity)
    st1 = (grid_idx * interval) - distparity - buffer
    end1 = ((grid_idx+1) * interval) - distparity + buffer
    st2 = st1 + distparity
    end2 = end1 + distparity
    st1 = 0 if st1 < 0 else st1
    st2 = 0 if st2 < 0 else st2
    if not left_first:
        temp = st1, end1
        st1, end1 = st2, end2
        st2, end2 = temp
    return st1, end1, st2, end2

def generate_grid_mask(img_src, row_st, row_end, col_st, col_end):
    mask = np.zeros_like(img_src)
    mask[row_st:row_end, col_st:col_end] = 1
    return mask

def find_tie_points_grids_matching(gray1, gray2, nfeatures=1000, topn_n_matches=300, grids=(1, 1), show_process=False):
    """split the image into grids to find the descriptors and match corresponding grid to ensure the well distributed tie points.
    
    Parameters
    ----------
    gray1: ndarray
        The first image with only one channel.
    gray2: ndarray
        The second image with only one channel.
    nfeatures: int
        The number of descriptor to find in one image.
    topn_n_matches: int, optional, default: 300
        The top n match for a single grid links to get the tie point pairs.
    grids: tuple of int, optional, default: (1, 1)
        The girdspec to split the image. The Length of grids should be 2, which 
        means (split_rows, split_cols).

    Returns
    -------
    kp1s_pts: ndarray
        The shape of kp1s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2s_pts: ndarray
        The shape of kp2s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    """
    # run first run tie points to get the median disparity
    kp1_pts, kp2_pts = find_tie_points_grids(gray1, gray2, nfeatures=4000, topn_n_matches=300, grids=(1, 1))
    med_x_dist = int(np.median(kp2_pts[:, 0] - kp1_pts[:, 0]))
    med_y_dist = int(np.median(kp2_pts[:, 1] - kp1_pts[:, 1]))

    # run matching for each grid
    buffer = 50 # to pad the grid to have more space for matching
    h, w = gray1.shape
    grid_h, grid_w = int((h-np.abs(med_y_dist))//grids[0]), int((w-np.abs(med_x_dist))//grids[1])
    kp1s_pts, kp2s_pts = [], []
    orb = cv2.ORB_create(nfeatures) # cv2.xfeatures2d.SURF_create(nfeatures) # cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    for grid_row_idx in range(grids[0]):
        for grid_col_idx in range(grids[1]):
            row_st1, row_end1, row_st2, row_end2 = generate_grid_idxs(med_y_dist, buffer, grid_row_idx, grid_h)
            col_st1, col_end1, col_st2, col_end2 = generate_grid_idxs(med_x_dist, buffer, grid_col_idx, grid_w)
            mask1 = generate_grid_mask(gray1, row_st1, row_end1, col_st1, col_end1)
            mask2 = generate_grid_mask(gray2, row_st2, row_end2, col_st2, col_end2)

            # find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(gray1, mask1)
            kp2, des2 = orb.detectAndCompute(gray2, mask2)
            if des1 is not None and des2 is not None:
                # match descriptors
                # des1, des2 = des1.astype(np.uint8), des2.astype(np.uint8)
                matches = bf.match(des1, des2) # Match descriptors.
                matches = sorted(matches, key=lambda x:x.distance)[:topn_n_matches]
                kp1_pts = np.array([kp.pt for kp in kp1])
                kp2_pts = np.array([kp.pt for kp in kp2])
                kp1_paired_idxs = [m.queryIdx for m in matches]
                kp2_paired_idxs = [m.trainIdx for m in matches]

                # convert Matches and kps into numpy array and list
                kp1_pts = np.array(kp1_pts)[kp1_paired_idxs]
                kp2_pts = np.array(kp2_pts)[kp2_paired_idxs]
                kp1s_pts.extend(kp1_pts)
                kp2s_pts.extend(kp2_pts)

                if show_process:
                    gray1_temp = gray1.copy()
                    gray2_temp = gray2.copy()
                    gray1_temp[~mask1.astype(np.bool)] = 0
                    gray2_temp[~mask2.astype(np.bool)] = 0
                    plot_kp_lines(gray1_temp, kp1_pts, gray2_temp, kp2_pts)


    kp1s_pts, kp2s_pts = np.array(kp1s_pts), np.array(kp2s_pts)
    return kp1s_pts, kp2s_pts


def find_tie_points_stereo_grids(gray1, gray2, nfeatures=1000, topn_n_matches=300, gms=False, grids=(1, 1)):
    """split the image into grids to find the descriptors to ensure the well distributed tie points.
    
    Parameters
    ----------
    gray1: ndarray
        The first image with only one channel.
    gray2: ndarray
        The second image with only one channel.
    nfeatures: int
        The number of descriptor to find in one image.
    topn_n_matches: int, optional, default: 300
        The top n match links to get the tie point pairs.
    grids: tuple of int, optional, default: (1, 1)
        The girdspec to split the image. The Length of grids should be 2, which 
        means (split_rows, split_cols).

    Returns
    -------
    kp1s_pts: ndarray
        The shape of kp1s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2s_pts: ndarray
        The shape of kp2s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    """
    h, w = gray1.shape[:2]
    grid_h, grid_w = int(h//grids[0]), int(w//grids[1])
    kp1s, kp2s, des1s, des2s = [], [], [], []
    for grid_row_idx in range(grids[0]):
        for grid_col_idx in range(grids[1]):
            # orb = cv2.xfeatures2d.SURF_create(nfeatures)
            orb = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers=10)
            # orb.setHessianThreshold(50000)
            # orb = cv2.ORB_create(nfeatures, scaleFactor=1.05, patchSize=101, WTA_K=4)
            row_st, row_end = grid_row_idx * grid_h, (grid_row_idx+1) * grid_h
            col_st, col_end = grid_col_idx * grid_w, (grid_col_idx+1) * grid_w
            mask = np.zeros_like(gray1)
            mask[row_st:row_end, col_st:col_end] = 1
            kp1, des1 = orb.detectAndCompute(gray1, mask)
            kp2, des2 = orb.detectAndCompute(gray2, mask)
            if des1 is not None:
                kp1s.extend(kp1); des1s.extend(des1);  
            if des2 is not None:
                kp2s.extend(kp2); des2s.extend(des2)            
    des1s = np.stack(des1s).astype(np.uint8)
    des2s = np.stack(des2s).astype(np.uint8)
    
    # match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1s, des2s) # Match descriptors.
    if gms:
        matches = cv2.xfeatures2d.matchGMS(gray1.shape[:2], gray2.shape[:2], kp1s, kp2s, matches, withScale=True, withRotation=True, thresholdFactor=2)
    matches = sorted(matches, key=lambda x:x.distance)

    # convert Matches and kps into numpy array and list
    kp1s_pts = np.array([kp.pt for kp in kp1s])
    kp2s_pts = np.array([kp.pt for kp in kp2s])
    kp1_paired_idxs = [m.queryIdx for m in matches[:topn_n_matches]]
    kp2_paired_idxs = [m.trainIdx for m in matches[:topn_n_matches]]
    kp1s_pts = np.array(kp1s_pts)[kp1_paired_idxs]
    kp2s_pts = np.array(kp2s_pts)[kp2_paired_idxs]
    return kp1s_pts, kp2s_pts

def find_tie_points_3imgs(gray1, gray2, gray3, mask1, mask2, mask3, nfeatures=1000, topn_n_matches=300):
    # orb = cv2.ORB_create(nfeatures, scaleFactor=2, patchSize=31, WTA_K=2)
    # orb = cv2.ORB_create(nfeatures, scaleFactor=1.05, patchSize=101, WTA_K=4)
    orb = cv2.xfeatures2d.SIFT_create(nfeatures)
    # orb = cv2.xfeatures2d.SURF_create(nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, mask1)
    kp2, des2 = orb.detectAndCompute(gray2, mask2)
    kp3, des3 = orb.detectAndCompute(gray3, mask3)
    assert des1 is not None, "No descriptor"
    assert des2 is not None, "No descriptor"
    assert des3 is not None, "No descriptor"

    # match descriptors
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()
    matches12 = bf.match(des1, des2) # Match descriptors.
    matches12 = np.array(sorted(matches12, key=lambda x:x.distance)[:topn_n_matches])
    matches23 = bf.match(des2, des3) # Match descriptors.
    matches23 = np.array(sorted(matches23, key=lambda x:x.distance)[:topn_n_matches])
    
    kp121_paired_idxs = [m.queryIdx for m in matches12]
    kp122_paired_idxs = [m.trainIdx for m in matches12]
    kp232_paired_idxs = [m.queryIdx for m in matches23]
    kp233_paired_idxs = [m.trainIdx for m in matches23]

    overlap = np.array(list(set(kp122_paired_idxs) & set(kp232_paired_idxs)))
    cand12 = kp122_paired_idxs
    sorter12 = np.argsort(cand12)
    overlap_idxs12 = sorter12[np.searchsorted(cand12, overlap, sorter=sorter12)]
    cand23 = kp232_paired_idxs
    sorter23 = np.argsort(cand23)
    overlap_idxs23 = sorter23[np.searchsorted(cand23, overlap, sorter=sorter23)]

    matches12 = matches12[overlap_idxs12]
    matches23 = matches23[overlap_idxs23]

    # convert Matches and kps into numpy array and list
    kp1s_pts = np.array([kp.pt for kp in kp1])
    kp2s_pts = np.array([kp.pt for kp in kp2])
    kp3s_pts = np.array([kp.pt for kp in kp3])
    kp1_paired_idxs = [m.queryIdx for m in matches12]
    kp2_paired_idxs = [m.trainIdx for m in matches12]
    kp3_paired_idxs = [m.trainIdx for m in matches23]
    kp1s_pts = np.array(kp1s_pts)[kp1_paired_idxs]
    kp2s_pts = np.array(kp2s_pts)[kp2_paired_idxs]
    kp3s_pts = np.array(kp3s_pts)[kp3_paired_idxs]
    assert len(kp1_paired_idxs) != 0, "no keypoints found"
    return kp1s_pts, kp2s_pts, kp3s_pts


def find_tie_points_masks(gray1, gray2, mask1, mask2, nfeatures=1000, topn_n_matches=300, gms=False):
    """split the image into grids to find the descriptors to ensure the well distributed tie points.
    
    Parameters
    ----------
    gray1: ndarray
        The first image with only one channel.
    gray2: ndarray
        The second image with only one channel.
    nfeatures: int
        The number of descriptor to find in one image.
    topn_n_matches: int, optional, default: 300
        The top n match links to get the tie point pairs.
    grids: tuple of int, optional, default: (1, 1)
        The girdspec to split the image. The Length of grids should be 2, which 
        means (split_rows, split_cols).

    Returns
    -------
    kp1s_pts: ndarray
        The shape of kp1s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2s_pts: ndarray
        The shape of kp2s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    """

    # orb = cv2.ORB_create(nfeatures, scaleFactor=2, patchSize=31, WTA_K=2)
    # orb = cv2.AKAZE_create(nfeatures)
    # orb = cv2.ORB_create(nfeatures, scaleFactor=1.05, patchSize=101, WTA_K=4)
    orb = cv2.xfeatures2d.SIFT_create(nfeatures)
    # orb = cv2.xfeatures2d.SURF_create(nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, mask1)
    kp2, des2 = orb.detectAndCompute(gray2, mask2)
    assert des1 is not None, "No descriptor"
    assert des2 is not None, "No descriptor"


    # match descriptors
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2) # Match descriptors.
    if gms:
        matches = cv2.xfeatures2d.matchGMS(gray1.shape[:2], gray2.shape[:2], kp1, kp2, matches, withScale=True, withRotation=True, thresholdFactor=2)
    matches = sorted(matches, key=lambda x:x.distance)

    # convert Matches and kps into numpy array and list
    kp1s_pts = np.array([kp.pt for kp in kp1])
    kp2s_pts = np.array([kp.pt for kp in kp2])
    kp1_paired_idxs = [m.queryIdx for m in matches[:topn_n_matches]]
    kp2_paired_idxs = [m.trainIdx for m in matches[:topn_n_matches]]
    kp1s_pts = np.array(kp1s_pts)[kp1_paired_idxs]
    kp2s_pts = np.array(kp2s_pts)[kp2_paired_idxs]
    return kp1s_pts, kp2s_pts


def find_tie_points_farest(gray1, gray2, nfeatures=6000, k_for_knn=20, topn_n_matches=10, dist_score_thres=500):
    """Find top 10 most similar tie points in both image. Then find the descriptors 
    farest from the centroid of "10 most similar tie points" iterately. Use the distance 
    between the descriptor and "10 most similar tie points" as feature vector to get its paired
    descriptor in another image.
    
    Parameters
    ----------
    gray1: ndarray
        The first image with only one channel.
    gray2: ndarray
        The second image with only one channel.
    nfeatures: int
        The number of descriptor to find in one image.
    k_for_knn:
        #TODO
    topn_n_matches: int, optional, default: 300
        The top n match links to get the tie point pairs.
    dist_score_thres: 
        #TODO

    Returns
    -------
    kp1_pts: ndarray
        The shape of kp1_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2_pts: ndarray
        The shape of kp2_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    """
    # get descriptors
    orb = cv2.ORB_create(nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2) # Match descriptors.
    matches = sorted(matches, key=lambda x:x.distance)
    
    # convert Matches and kps into numpy array and list
    kp1_pts = np.array([kp.pt for kp in kp1])
    kp2_pts = np.array([kp.pt for kp in kp2])
    kp1_paired_idxs = [m.queryIdx for m in matches[:10]]
    kp2_paired_idxs = [m.trainIdx for m in matches[:10]]

    # get top k matches from knn matches 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_knn = bf.knnMatch(des1, des2, k=k_for_knn) # Match descriptors.
    matches_knn_rows = []
    for m in matches_knn:
        matches_knn_row = [m[0].queryIdx]
        matches_knn_row.extend([m[i].trainIdx for i in range(k_for_knn)])
        matches_knn_rows.append(matches_knn_row)
    columns = ['kp1']
    kp2_columns = ['kp2-'+str(i) for i in range(k_for_knn)]
    columns.extend(kp2_columns)
    df_matches_knn = pd.DataFrame(matches_knn_rows, columns=columns)

    # get top 10 matchesto calculate the centroid and find the farest descriptor 
    centorid = np.mean(kp1_pts[kp1_paired_idxs], axis=0)
    kp1_idxs_sort = np.argsort(pairwise_distances([centorid], kp1_pts)[0])[::-1]
    kp1_idxs_sort_last = np.array([kp1_idx for kp1_idx in kp1_idxs_sort if kp1_idx not in kp1_paired_idxs])
    
    # init lists for forloop to append
    kp1_paired_pts = list(kp1_pts[kp1_paired_idxs])
    kp2_paired_pts = list(kp2_pts[kp2_paired_idxs])
    final_paired_idxs = [(m.queryIdx, m.trainIdx) for m in matches[:topn_n_matches]]

    # for loop by nfeatures
    for i in range(nfeatures - topn_n_matches):
        query_idx = kp1_idxs_sort_last[i]
        train_idx_cands = df_matches_knn.loc[df_matches_knn['kp1'] == query_idx, kp2_columns].values[0]
        target_dist_pairs = pairwise_distances([kp1_pts[query_idx]], kp1_paired_pts)[0]

        # for loop by top k neightbors
        for train_idx_cand_idx in range(len(train_idx_cands)):
            test_dist_pairs = pairwise_distances([kp2_pts[train_idx_cands[train_idx_cand_idx]]], kp2_paired_pts)[0]
            dist_score = pairwise_distances([target_dist_pairs], [test_dist_pairs])[0][0]
            if dist_score < dist_score_thres:
                break

        # filter the point with different pattern
        if dist_score < dist_score_thres:
            final_paired_idxs.append((query_idx, train_idx_cands[train_idx_cand_idx]))
            kp1_paired_pts.append(kp1_pts[query_idx])
            kp2_paired_pts.append(kp2_pts[train_idx_cands[train_idx_cand_idx]])

    kp1_pts, kp2_pts = np.array(kp1_paired_pts), np.array(kp2_paired_pts)
    return kp1_pts, kp2_pts

def plot_kp_lines(img1_rgb, kp1_pts, img2_rgb, kp2_pts, title=None, texts_left=None, texts_right=None, fontsize=8, colors=None, figsize=(30, 30), ax=None):
    no_ax = ax is None
    img = np.concatenate([img1_rgb, img2_rgb], axis=1)
    kp2_pts = np.array(kp2_pts)
    kp2_pts[:, 0] += img1_rgb.shape[1]

    if no_ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if len(img1_rgb.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    
    ax.scatter(kp1_pts[:, 0], kp1_pts[:, 1], s=3, c='red')
    ax.scatter(kp2_pts[:, 0], kp2_pts[:, 1], s=3, c='red')

    for idx, (kp1_pt, kp2_pt) in enumerate(zip(kp1_pts, kp2_pts)):
        c = colors[idx] if colors is not None else np.random.rand(3)
        ax.plot((kp1_pt[0], kp2_pt[0]), (kp1_pt[1], kp2_pt[1]), c=c, linewidth=0.5)
        if texts_left is not None:
            ax.text(kp1_pt[0]-100, kp1_pt[1], texts_left[idx], fontsize=fontsize)
        if texts_right is not None:
            ax.text(kp2_pt[0], kp2_pt[1], texts_right[idx], fontsize=fontsize)


    if no_ax:
        if title is not None:
            plt.title(title)
        # plt.tight_layout()
        plt.show()

def plot_kp_lines_img1(img1_rgb, kp1_pts, kp2_pts, colors=None, figsize=(30, 30)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if len(img1_rgb.shape) == 2:
        ax.imshow(img1_rgb, cmap='gray')
    else:
        ax.imshow(img1_rgb)
    
    ax.scatter(kp1_pts[:, 0], kp1_pts[:, 1], s=3, c='red')

    for idx, (kp1_pt, kp2_pt) in enumerate(zip(kp1_pts, kp2_pts)):
        c = colors[idx] if colors is not None else np.random.rand(3)
        ax.plot((kp1_pt[0], kp2_pt[0]), (kp1_pt[1], kp2_pt[1]), c=c, linewidth=0.5)
    plt.show()


def plot_kp_lines_3imgs(img1_rgb, kp1_pts, img2_rgb, kp2_pts, img3_rgb, kp3_pts, colors=None, figsize=(20, 20)):
    img12 = np.concatenate([img1_rgb, img2_rgb], axis=1)
    img34 = np.concatenate([np.zeros_like(img3_rgb), img3_rgb], axis=1)
    img = np.concatenate([img12, img34], axis=0)

    kp2_pts = np.array(kp2_pts)
    kp2_pts[:, 0] += img1_rgb.shape[1]

    kp3_pts = np.array(kp3_pts)
    kp3_pts[:, 0] += img1_rgb.shape[1]
    kp3_pts[:, 1] += img1_rgb.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if len(img1_rgb.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    
    ax.scatter(kp1_pts[:, 0], kp1_pts[:, 1], s=3, c='red')
    ax.scatter(kp2_pts[:, 0], kp2_pts[:, 1], s=3, c='red')
    ax.scatter(kp3_pts[:, 0], kp3_pts[:, 1], s=3, c='red')

    for idx, (kp1_pt, kp2_pt, kp3_pt) in enumerate(zip(kp1_pts, kp2_pts, kp3_pts)):
        c = colors[idx] if colors is not None else np.random.rand(3)
        ax.plot((kp1_pt[0], kp2_pt[0]), (kp1_pt[1], kp2_pt[1]), c=c, linewidth=0.5)
        ax.plot((kp2_pt[0], kp3_pt[0]), (kp2_pt[1], kp3_pt[1]), c=c, linewidth=0.5)
        # if texts_left is not None:
        #     ax.text(kp1_pt[0]-100, kp1_pt[1], texts_left[idx], fontsize=fontsize)
        # if texts_right is not None:
        #     ax.text(kp2_pt[0], kp2_pt[1], texts_right[idx], fontsize=fontsize)

    plt.show()

def get_dist_from_tie_points(aereo_params1, aereo_params2, kp1_pts, kp2_pts):
    kp1_pts_npidxs = np.array([kp1_pts[:, 1], kp1_pts[:, 0]]).T
    kp2_pts_npidxs = np.array([kp2_pts[:, 1], kp2_pts[:, 0]]).T
    dep1, dep2, pxyz1, pxyz2, dists_ecu, dists_blk = get_PQ(aereo_params1, aereo_params2, kp1_pts_npidxs, kp2_pts_npidxs, return_depth=True)
    return dep1, dep2, pxyz1, pxyz2, dists_ecu

def filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=10, max_depth=None, median_dist=False):
    """Filter the tie points by the minimum distance of image rays (projection center & object 
    location) of both images. The minimum distance of image rays is calculated from aerotriangulation.
    As a result, aereo_params1 (location and rotation) must have a certain degree of accuracy.

    Parameters
    ----------
    aereo_params1: list
        [OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE].
    aereo_params2: list
        [OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE].
    kp1_pts: ndarray
        The shape of kp1_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2_pts: ndarray
        The shape of kp2_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    max_dist: int, default: 10
        The max acceptable minimum distance of image rays. Its unit is meters.

    Returns
    -------
    kp1s_pts: ndarray
        The shape of kp1s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    kp2s_pts: ndarray
        The shape of kp2s_pts is (None, 2). The first dimention means the 
        number of tie points. The second dimention means the x and y  (not row and 
        column indices) of the tie point.
    """
    dep1, dep2, pxyz1, pxyz2, dists_ecu = get_dist_from_tie_points(aereo_params1, aereo_params2, kp1_pts, kp2_pts)
    dists_ecu = dists_ecu - np.median(dists_ecu) if median_dist else dists_ecu
    valid_pt_mask = (np.abs(dists_ecu) < max_dist) & (dep1 > 0)
    if max_depth is not None:
        valid_pt_mask &= (dep1 < max_depth)
    valid_pt_idxs = np.where(valid_pt_mask)
    dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = dep1[valid_pt_idxs], dep2[valid_pt_idxs], pxyz1[valid_pt_idxs], pxyz2[valid_pt_idxs], kp1_pts[valid_pt_idxs], kp2_pts[valid_pt_idxs], dists_ecu[valid_pt_idxs]
    return dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu

def filter_tie_points_by_row_diff(kp1_pts, kp2_pts, max_diff=10):
    row_diff = kp1_pts[:, 1] - kp2_pts[:, 1]
    mask = [d <= max_diff for d in row_diff]
    return kp1_pts[mask], kp2_pts[mask], mask

def plot_aero_triangulation(aereo_params1, aereo_params2, pxyz1, pxyz2, ax=None, colors=None, scale_img=10, title='AeroTriangulation of Both Images'):
    """
    create your own ax:
        ```
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ```
    """
    no_ax = ax is None
    from mpl_toolkits.mplot3d import Axes3D
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET1, YOFFSET1 = aereo_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2, XOFFSET2, YOFFSET2 = aereo_params2

    npidxs = [(0, 0), (0, COLS1), (ROWS1, COLS1), (ROWS1, 0)]
    P_XYZs1 = project_npidxs_to_XYZs_by_k(npidxs, aereo_params1, k=1/scale_img)
    P_XYZs2 = project_npidxs_to_XYZs_by_k(npidxs, aereo_params1, k=1/scale_img)

    if no_ax:
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    XYZ1 = (P_XYZs1[[0,1,3,2], :].reshape(2, 2, -1).transpose(2, 0, 1) / 1000) * scale_img
    XYZ2 = (P_XYZs2[[0,1,3,2], :].reshape(2, 2, -1).transpose(2, 0, 1) / 1000) * scale_img
    ax.plot_surface(*XYZ1, alpha=0.3)
    ax.plot_surface(*XYZ2, alpha=0.3)

    for idx, (p1, p2) in enumerate(zip(pxyz1, pxyz2)):
        c = colors[idx] if colors is not None else np.random.rand(3)
        ax.plot([L_XYZ1[0], p1[0]], [L_XYZ1[1], p1[1]], zs=[L_XYZ1[2], p1[2]], color=c, linewidth=1)
        ax.plot([L_XYZ2[0], p1[0]], [L_XYZ2[1], p2[1]], zs=[L_XYZ2[2], p2[2]], color=c, linewidth=1)

    ax.scatter([L_XYZ1[0]], [L_XYZ1[1]], [L_XYZ1[2]], color='red')
    ax.scatter([L_XYZ2[0]], [L_XYZ2[1]], [L_XYZ2[2]], color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    all_ps = np.concatenate([pxyz1, pxyz2, [L_XYZ1], [L_XYZ2]])
    axs, ays, azs = all_ps.T
    bufferx, buffery, bufferz = (axs.max() - axs.min()) * 0.03, (ays.max() - ays.min()) * 0.03, (azs.max() - azs.min()) * 0.03
    ax.set_xlim(axs.min()-bufferx, axs.max()+bufferx)
    ax.set_ylim(ays.min()-buffery, ays.max()+buffery)
    ax.set_zlim(azs.min()-bufferz, azs.max()+bufferz)

    if no_ax:
        plt.show()


def plot_aero_triangulation_points(aereo_params1, aereo_params2, pxyz1, pxyz2, ax=None, colors=None, scale_img=10, title='AeroTriangulation of Both Images'):
    """
    create your own ax:
        ```
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ```
    """
    no_ax = ax is None
    from mpl_toolkits.mplot3d import Axes3D
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET1, YOFFSET1 = aereo_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2, XOFFSET2, YOFFSET2 = aereo_params2

    npidxs = [(0, 0), (0, COLS1), (ROWS1, COLS1), (ROWS1, 0)]
    P_XYZs1 = project_npidxs_to_XYZs_by_k(npidxs, aereo_params1, k=1/scale_img)
    P_XYZs2 = project_npidxs_to_XYZs_by_k(npidxs, aereo_params1, k=1/scale_img)

    if no_ax:
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    XYZ1 = (P_XYZs1[[0,1,3,2], :].reshape(2, 2, -1).transpose(2, 0, 1) / 1000) * scale_img
    XYZ2 = (P_XYZs2[[0,1,3,2], :].reshape(2, 2, -1).transpose(2, 0, 1) / 1000) * scale_img
    ax.plot_surface(*XYZ1, alpha=0.3)
    ax.plot_surface(*XYZ2, alpha=0.3)

    for idx, (p1, p2) in enumerate(zip(pxyz1, pxyz2)):
        c = colors[idx] if colors is not None else np.random.rand(3)
        ax.scatter([p1[0]], [p1[1]], [p1[2]], color=c)
        ax.scatter([p2[0]], [p2[1]], [p2[2]], color=c)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if no_ax:
        plt.show()



def plot_aero_triangulation_multi_imgs(aereo_params_imgs, pxyz_imgs, ax=None, colors=None, scale_img=10, title='AeroTriangulation of Both Images'):
    """
    create your own ax:
        ```
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ```
    """
    from mpl_toolkits.mplot3d import Axes3D
    no_ax = ax is None
    if no_ax:
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(title)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    if colors is None:
        colors = [np.random.rand(3) for i in pxyz_imgs[0]]

    for aereo_params, pxyz in zip(aereo_params_imgs, pxyz_imgs):
        OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE, XOFFSET, YOFFSET = aereo_params
        npidxs = [(0, 0), (0, COLS), (ROWS, COLS), (ROWS, 0)]
        P_XYZs = project_npidxs_to_XYZs_by_k(npidxs, aereo_params, k=1/scale_img)
        XYZ = (P_XYZs[[0,1,3,2], :].reshape(2, 2, -1).transpose(2, 0, 1) / 1000) * scale_img

        ax.plot_surface(*XYZ, alpha=0.3)
        ax.scatter([L_XYZ[0]], [L_XYZ[1]], [L_XYZ[2]], color='red')
        for idx, p in enumerate(pxyz):
            ax.plot([L_XYZ[0], p[0]], [L_XYZ[1], p[1]], zs=[L_XYZ[2], p[2]], color=colors[idx], linewidth=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if no_ax:
        plt.show()

if __name__ =='__main__':
    import os 
    import TronGisPy as tgp
    from io_aereo_params import get_DMC_aereo_params

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
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # find tie points
    # kp1_pts, kp2_pts = find_tie_points_grids(gray1, gray2, nfeatures=1000, topn_n_matches=100, grids=(3, 3))
    kp1_pts, kp2_pts = find_tie_points_grids_matching(gray1, gray2, nfeatures=1000, topn_n_matches=100, grids=(3, 3), show_process=True)
    x_diff = kp1_pts[:, 0] - kp2_pts[:, 0]
    plot_kp_lines(img1, kp1_pts, img2, kp2_pts, title="TiePoints", figsize=(10, 10), texts_left=["%.2f"%d for d in x_diff])

    # ks1, ks2, pxyz1, pxyz2, dists_ecu = get_dist_from_tie_points(aereo_params1, aereo_params2, kp1_pts, kp2_pts)
    # plt.hist(dists_ecu)
    # plt.title("Distance Stats in Object Space")
    # plt.xlabel('distance(m)')
    # plt.ylabel('count of tie points pairs')
    # plt.show()

    # filter tie points
    dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=3)
    # texts_left = ["d=%.2f, "%d + "k=%.2f"%k for idx, (d, k) in enumerate(zip(dists_ecu, dep1))]
    # plot_kp_lines(img1, kp1_pts, img2, kp2_pts, texts_left=texts_left, fontsize=12)
    plot_kp_lines(img1, kp1_pts, img2, kp2_pts, title="TiePoints (Filtered)")
    print(len(kp1_pts))

