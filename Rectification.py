import cv2
import numpy as np
from matplotlib import pyplot as plt

def is_vector(x):
	dimension = len(x.shape)
	if (dimension == 1): return True
	if (dimension == 2):
		if (x.shape[0] == 1) or (x.shape[1] == 1):
			return True
		return False
	return False

def homogeneous_to_euclidean(x):
	"""
	See: OpenCV - convertPointsFromHomogeneous
	The function converts points homogeneous to Euclidean space using perspective projection.
	That is, each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn).
	When xn=0, the output point coordinates will be (0,0,0,...).
	"""
	# Faster method: avoid dividing everything
	y = x.astype(np.float32) # Make a copy of x and convert to FLOATS
	if is_vector(y):
		y[-1] = np.nan if (y[-1] == 0) else y[-1]
		y[:-1] /= y[-1]
		y[np.isnan(y)] = 0
		y[-1] = 0 if (y[-1] == 0) else 1
	else:
		y[-1, y[-1] == 0] = np.nan	# Replace 0s in end row with nan to prevent division by zero
		y[:-1] /= y[-1]					# Divide all rows excluding end row, by end row
		y[np.isnan(y)] = 0			# Replace all nan with 0s
		y[-1, y[-1] != 0] = 1			# Replace end row with 1s if the value is not zero
	return y

def rectify_shearing(H1, H2, image_width, image_height):
	##### ##### ##### ##### ##### 
	##### CREDIT
	##### ##### ##### ##### ##### 

	# Loop & Zhang - via literature
	#	* http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
	# TH. - via stackexchange user
	# 	* http://scicomp.stackexchange.com/users/599/th
	#	* http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification

	##### ##### ##### ##### ##### 
	##### PARAMETERS
	##### ##### ##### ##### ##### 

	# Let H1 be the rectification homography of image1 (ie. H1 is a homogeneous space)
	# Let H2 be the rectification homography of image2 (ie. H2 is a homogeneous space)
	# image_width, image_height be the dimensions of both image1 and image2

	##### ##### ##### ##### ##### 

	"""
	Compute shearing transform than can be applied after the rectification transform to reduce distortion.
	Reference:
		http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
		"Computing rectifying homographies for stereo vision" by Loop & Zhang
	"""

	w = image_width
	h = image_height

	'''
	Loop & Zhang use a shearing transform to reduce the distortion
	introduced by the projective transform that mapped the epipoles to infinity
	(ie, that made the epipolar lines parallel).

	Consider the shearing transform:

			| k1 k2 0 |
	S	=	| 0  1  0 |
			| 0  0  1 |

	Let w and h be image width and height respectively.
	Consider the four midpoints of the image edges:
	'''

	a = np.float32([ (w-1)/2.0,	0,			1 ])
	b = np.float32([ (w-1),		(h-1)/2.0,	1 ])
	c = np.float32([ (w-1)/2.0,	(h-1),		1 ])
	d = np.float32([ 0,			(h-1)/2.0,	1 ])

	'''
	According to Loop & Zhang:
	"... we attempt to preserve perpendicularity and aspect ratio of the lines bd and ca"
	'''

	'''
	Let H be the rectification homography and,
	Let a' = H*a be a point in the affine plane by dividing through so that a'2 = 1
	Note: a'2 is the third component, ie, a' = (a'[0], a'1, a'2))
	'''

	# Note: *.dot is a form of matrix*vector multiplication in np
	# So a_prime = H*a such that a_prime[2] = 1 (hence the use of homogeneous_to_euclidean function)

	a_prime = homogeneous_to_euclidean(H1.dot(a))
	b_prime = homogeneous_to_euclidean(H1.dot(b))
	c_prime = homogeneous_to_euclidean(H1.dot(c))
	d_prime = homogeneous_to_euclidean(H1.dot(d))

	''' Let x = b' - d' and y = c' - a' '''

	x = b_prime - d_prime
	y = c_prime - a_prime

	'''
	According to Loop & Zhang:
		"As the difference of affine points, x and y are vectors in the euclidean image plane.
			Perpendicularity is preserved when (Sx)^T(Sy) = 0, and aspect ratio is preserved if [(Sx)^T(Sx)]/[(Sy)^T(Sy)] = (w^2)/(h^2)"
	'''

	''' The real solution presents a closed-form: '''

	k1 = (h*h*x[1]*x[1] + w*w*y[1]*y[1]) / (h*w*(x[1]*y[0] - x[0]*y[1]))
	k2 = (h*h*x[0]*x[1] + w*w*y[0]*y[1]) / (h*w*(x[0]*y[1] - x[1]*y[0]))

	''' Determined by sign (the positive is preferred) '''

	if (k1 < 0): # Why this?
		k1 *= -1
		k2 *= -1

	return np.float32([[k1,  k2,  0], [0,  1,  0], [0,  0,  1]])

def get_rectify_param(img_shape, kp1_pts, kp2_pts, K=np.eye(3), d=None, shearing=True):
    """Compute Fundamental matrix

    Parameters
    ----------
    # K		Camera matrix
    # d		Distortion parameters
    # kp1_pts	Feature points in image1
    # kp2_pts	Corresponding feature points in image2
    # F		Fundamental matrix
    # H1	Homography matrix transform for image1
    # H2	Homography matrix transform for image2
    # R1	Rectification matrix transform for image1
    # R2	Rectification matrix transform for image2
    """
    # Get Fundamental matrix: Rectification based on found Fundamental matrix
    F, F_mask = cv2.findFundamentalMat(kp1_pts, kp2_pts)
    F_mask = F_mask.flatten() # Select only inlier points (filter outliers)
    kp1_pts = kp1_pts[F_mask == 1]
    kp2_pts = kp2_pts[F_mask == 1]

    # Calculate Homogeneous matrix transform given features and fundamental matrix
    image_size = (img_shape[1], img_shape[0])
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(kp1_pts, kp2_pts, F, image_size) # Note: image_size is not image_shape

    if (retval == False):
        print("ERROR: stereoRectifyUncalibrated failed")
        return None

    # Apply a shearing transform to homography matrices
    if shearing:
        S = rectify_shearing(H1, H2, *image_size)
        H1 = S.dot(H1)
	
    # Compute the rectification transform
    K_inverse = np.linalg.inv(K)
    R1 = K_inverse.dot(H1).dot(K)
    R2 = K_inverse.dot(H2).dot(K)

    mapx1, mapy1 = cv2.initUndistortRectifyMap(K, d, R1, K, image_size, cv2.CV_16SC2)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(K, d, R2, K, image_size, cv2.CV_16SC2)

    # fig, axes = plt.subplots(2, 3 , figsize=(15, 5))
    # axes = axes.flatten()
    # axes[0].imshow(mapx1[:, :, 0], cmap='gray')
    # axes[1].imshow(mapx1[:, :, 1], cmap='gray')
    # axes[2].imshow(mapy1, cmap='gray')
    # axes[3].imshow(mapx2[:, :, 0], cmap='gray')
    # axes[4].imshow(mapx2[:, :, 1], cmap='gray')
    # axes[5].imshow(mapy2, cmap='gray')
    # plt.show()
    return H1, H2, mapx1, mapy1, mapx2, mapy2

def drawlines(img1, img2, lines, pts1, pts2, colors=None):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img1, img2 = img1.copy(), img2.copy()
    r, c = img1.shape[:2]
    pts1 = pts1.astype(np.int)
    pts2 = pts2.astype(np.int)
    for idx, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = np.random.rand(3) if colors is None else colors[idx]
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 2, cv2.LINE_AA)
        img1 = cv2.circle(img1, tuple(pt1), 3, color, 30)
        img2 = cv2.circle(img2, tuple(pt2), 3, color, 30)
    return img1, img2

def plot_epipolar(img1, img2, kp1_pts, kp2_pts, plot_size=None):
    if img1.ndim == 2:
        img1 = np.stack([img1, img1, img1]).transpose(1, 2, 0) / 255
        img2 = np.stack([img2, img2, img2]).transpose(1, 2, 0) / 255
    F, F_mask = cv2.findFundamentalMat(kp1_pts, kp2_pts)

    if plot_size is not None:
        rand_idxs = np.random.choice(range(len(kp1_pts)), size=plot_size)
        kp1_pts, kp2_pts = kp1_pts[rand_idxs], kp2_pts[rand_idxs]
    colors = np.random.rand(len(kp1_pts), 3)
    colors = colors * 255 if img1.dtype == np.uint8 else colors

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(kp2_pts.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img1, img2, lines1, kp1_pts, kp2_pts, colors=colors)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(kp1_pts.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(img2, img1, lines2, kp2_pts, kp1_pts, colors=colors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    plt.suptitle('Epipolar Lines')
    ax1.imshow(img5)
    ax1.set_title('left image')
    ax2.imshow(img3)
    ax2.set_title('right image')
    plt.show()

def rectify(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, border_value=-1):
    """Apply Rectification Transform

    Parameters
    ----------
    interpolation: {cv2.INTER_CUBIC, cv2.INTER_LINEAR}

    """
    rectified = cv2.remap(image, mapx, mapy, 
                        interpolation=interpolation,  
                        borderMode=cv2.BORDER_CONSTANT, 
                        borderValue=border_value)
    return rectified

def rectify_kps(kps, H):
    """
    kps = [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    kps = np.hstack([kps, np.ones((len(kps), 1))])
    kps_rectified = np.dot(H, kps.T).T
    kps_rectified /= np.tile(kps_rectified[:, 2], (3, 1)).T
    return kps_rectified[:, :2]

def unrectify_kps(kps_rectified, H):
    """
    kps_rectified = [(x1, y1), (x2, y2), (x3, y3), ...]
    """
    kps_rectified = np.hstack([kps_rectified, np.ones((len(kps_rectified), 1))])
    kps_ori = np.dot(np.linalg.inv(H), kps_rectified.T).T
    kps_ori /= np.tile(kps_ori[:, 2], (3, 1)).T
    return kps_ori[:, :2]

def rectify_idxs(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, border_value=-1):
    rows, cols = image.shape[:2]
    image1_ci, image1_ri = np.meshgrid(range(cols), range(rows))
    image1_xys = np.stack([image1_ci, image1_ri]).astype(np.int32).transpose(1, 2, 0)
    rectified_xys = rectify(image1_xys, mapx, mapy, interpolation=cv2.INTER_NEAREST)
    return rectified_xys

def plot_rectified_img(rectified1, rectified2, line_interval=100):
    if len(rectified1.shape) == 3:
        rows, cols, bands = rectified1.shape
        img = np.empty((rows, cols*2, bands), dtype=rectified1.dtype)
    else:
        rows, cols = rectified1.shape
        img = np.empty((rows, cols*2), dtype=rectified1.dtype)
    img[:rows, :cols] = rectified1.copy()
    img[:rows, cols:] = rectified2.copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_title("Rectification Result")
    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    for y in np.arange(0, rows, line_interval):
        ax.axhline(y=y, color=np.random.rand(3), linestyle='-', linewidth=0.5)
    plt.show()

def plot_rectified_img2(rectified1, rectified2):
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].imshow(cv2.addWeighted(rectified1, 0.5, rectified2, 0.5, 0))
    axes[1].imshow(cv2.addWeighted(rectified1, 1, rectified2, -1, 0))
    plt.show()


if __name__ =='__main__':
    import os 
    import cv2
    import numpy as np
    import TronGisPy as tgp
    from matplotlib import pyplot as plt
    from io_aereo_params import get_DMC_aereo_params
    from TiePoints import find_tie_points_grids, plot_kp_lines, filter_tie_points_by_PQ_dist, find_tie_points_grids_matching

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

    # find tie points & filter tie points
    kp1_pts, kp2_pts = find_tie_points_grids_matching(gray1, gray2, nfeatures=4000, topn_n_matches=300, grids=(3, 3)) # find_tie_points_farest(gray1, gray2)
    dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=5)

    # rectify image
    H1, H2, mapx1, mapy1, mapx2, mapy2 = get_rectify_param(gray1.shape, kp1_pts, kp2_pts, shearing=True)
    rectified1_norm = rectify(img1, mapx1, mapy1, border_value=0)
    rectified2_norm = rectify(img2, mapx2, mapy2, border_value=0)
    plot_epipolar(img1, img2, kp1_pts, kp2_pts)
    plot_rectified_img(rectified1_norm, rectified2_norm)
    plot_rectified_img2(rectified1_norm, rectified2_norm)

    # rectify image index
    rectified_npidxs1 = rectify_idxs(img1, mapx1, mapy1, border_value=-1)
    rectified_npidxs2 = rectify_idxs(img2, mapx2, mapy2, border_value=-1)
    rectified_npidxs1_show = tgp.Normalizer().fit_transform(rectified_npidxs1[:, :, 0] * rectified_npidxs1[:, :, 1])
    rectified_npidxs2_show = tgp.Normalizer().fit_transform(rectified_npidxs2[:, :, 0] * rectified_npidxs2[:, :, 1])

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(rectified_npidxs1_show, cmap='gray')
    # ax2.imshow(rectified_npidxs2_show, cmap='gray')
    # plt.show()


