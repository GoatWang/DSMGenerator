import numpy as np

def get_M(OPK):
    """Collinearity Equation"""
    omega, phi, kappa = OPK[0], OPK[1], OPK[2]
    cO, sO = np.cos(omega), np.sin(omega)
    cP, sP = np.cos(phi), np.sin(phi)
    cK, sK = np.cos(kappa), np.sin(kappa)
    M = np.stack([[  cP*cK, cO*sK+sO*sP*cK, sO*sK-cO*sP*cK],
                  [ -cP*sK, cO*cK-sO*sP*sK, sO*cK+cO*sP*sK],
                  [     sP,         -sO*cP,          cO*cP]])
    return M

def convert_npidxs_to_imxys(idxs, rows=13824, cols=7680, pixel_size=0.012, x_offset=0, y_offset=0):
    """
    image space linear transform
    pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    idxs
    """
    col_zoomout, row_zoomout = pixel_size, -pixel_size # negative because inverted dirrection (up=>+, down=>-) of y-axis 
    idxs_trans = np.transpose(idxs)
    row_idxs, col_idxs = idxs_trans[0, :], idxs_trans[1, :]
    xs = ((col_idxs - (cols/2)) * col_zoomout) + x_offset
    ys = ((row_idxs - (rows/2)) * row_zoomout) + y_offset
    imxys = np.transpose(np.stack([xs, ys]))
    return imxys

def convert_imxyzs_to_npidxs(imxyzs, rows=13824, cols=7680, pixel_size=0.012, x_offset=0, y_offset=0):
    """
    image space linear transform
    pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    """
    col_zoomin, row_zoomin = 1/pixel_size, -1/pixel_size
    xs, ys = imxyzs.T[0], imxyzs.T[1]
    col_idxs = (col_zoomin * (xs - x_offset)) + (cols/2)
    row_idxs = (row_zoomin * (ys - y_offset)) + (rows/2)
    npidxs = np.array([row_idxs, col_idxs]).T
    return npidxs

def project_npidxs_to_XYZs(P_npidxs, P_Z, aerotri_params, return_k=False):
    """
    idxs conversion from image space and object space.
    P_xyzs: xyzs of points in image space.
    P_Z: assume height.
    return_k: assume height.
    """
    # aerotri_params
    OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE, XOFFSET, YOFFSET = aerotri_params
    # L_XYZ = np.expand_dims(L_XYZ, axis=-1)

    # calculate P_imxyzs from P_npidxs
    P_xys = convert_npidxs_to_imxys(np.array(P_npidxs), ROWS, COLS, PIXEL_SIZE, XOFFSET, YOFFSET) # (n points, 2 dim)
    P_xyzs = np.concatenate([P_xys, np.full((len(P_xys), 1), -FOCAL_LENGTH)], axis=1).T/1000 # add -focal_length as new dim to (3 dim, n points)

    # calculate P_XYZs from P_imxyzs
    M, L_Z = get_M(OPK), L_XYZ[2]
    mt_xyzs = np.matmul(M.T, P_xyzs)
    ks = (mt_xyzs[2]) / (P_Z - L_Z) # convert from (L_Z - P_Z) = P_xyzs[2] / k, and query for k, /1000 mm => m
    P_XYZs = ((mt_xyzs / ks) + np.expand_dims(L_XYZ, axis=-1)).T # 4 points, 3 dim
    if not return_k:
        return P_XYZs
    else:
        return P_XYZs, ks

def project_npidxs_to_XYZs_by_k(P_npidxs, aerotri_params, k=1):
    """
    idxs conversion from image space and object space.
    P_xyzs: xyzs of points in image space.
    P_Z: assume height.
    return_k: assume height.
    """
    # aerotri_params
    OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE, XOFFSET, YOFFSET = aerotri_params
    # L_XYZ = np.expand_dims(L_XYZ, axis=-1)

    # calculate P_imxyzs from P_npidxs
    P_xys = convert_npidxs_to_imxys(np.array(P_npidxs), ROWS, COLS, PIXEL_SIZE, XOFFSET, YOFFSET) # (n points, 2 dim)
    P_xyzs = np.concatenate([P_xys, np.full((len(P_xys), 1), -FOCAL_LENGTH)], axis=1).T/1000 # add -focal_length as new dim to (3 dim, n points), /1000 mm => m

    # calculate P_XYZs from P_imxyzs
    M, L_Z = get_M(OPK), L_XYZ[2]
    mt_xyzs = np.matmul(M.T, P_xyzs)
    P_XYZs = ((mt_xyzs) + np.expand_dims(L_XYZ, axis=-1)).T # 4 points, 3 dim
    return P_XYZs

def project_XYZs_to_npidxs(P_XYZs, aerotri_params, return_k=False):
    """
    idxs conversion from object space and image space.
    P_XYZs: XYZs of points in object space.
    aerotri_params: opk, L_XYZ, rows, cols, focal_length, pixel_size
        opk: omega, phi, kappa.
        L_XYZ: the location of the camera.
        focal_length: dmc image is 120 (mm).
        pixel_size: dmc image is sensor_width/image_cols = sensor_height/image_rows = 0.012 (mm).
    """
    # aerotri_params
    OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE, XOFFSET, YOFFSET = aerotri_params
    
    # calculate P_imxyzs from P_XYZs
    M = get_M(OPK)
    P_XYZs = np.array(P_XYZs).T  # =>P_XYZs.shape==(3, None)
    L_XYZ = np.array(L_XYZ).reshape(3, 1) # =>L_XYZ.shape==(3, 1)
    dXYZs = (P_XYZs-L_XYZ) # =>xyz.shape==(3, None)
    m_XYZs = np.matmul(M, dXYZs)
    ks = -(FOCAL_LENGTH/1000) / m_XYZs[2] # convert FOCAL_LENGTH from mm to m
    P_xyzs = (ks * m_XYZs).T # (n points, 3 dim)
    
    # calculate P_npidxs from P_xyzs
    P_npidxs = convert_imxyzs_to_npidxs(P_xyzs, ROWS, COLS, PIXEL_SIZE, XOFFSET, YOFFSET)
    if not return_k:
        return P_npidxs
    else:
        return P_npidxs, ks

def get_line_vecs(idxs, opk, lxyz, rows=13824, cols=7680, pixel_size=0.012, focal_length=120, x_offset=0, y_offset=0):
    P_xys = convert_npidxs_to_imxys(idxs, rows=rows, cols=cols, pixel_size=pixel_size, x_offset=x_offset, y_offset=y_offset) # (n points, 2 dim)
    focal_lengths = np.ones((np.shape(P_xys)[0], 1)) * -focal_length
    P_xyzs = np.transpose(np.concatenate([P_xys, focal_lengths], axis=1)) # add -focal_length as new dim to (3 dim, n points)
    M = get_M(opk)
    vecs = np.transpose(np.matmul(np.transpose(M), P_xyzs))
    return M, vecs

# def get_line_vecs_camera_matrix(idxs, opk, lxyz, px=0, py=0, rows=13824, cols=7680, pixel_size=0.012, focal_length=120):
#     P_xys = convert_npidxs_to_imxys(idxs, rows=rows, cols=cols, pixel_size=pixel_size) # (n points, 2 dim)
#     f = focal_length
#     sx, sy = pixel_size, pixel_size
#     camera_matrix = np.array([[ f/sx,    0, cols/2+px/sx],
#                               [    0, f/sy, rows/2+py/sy],
#                               [    0,    0,            1]], dtype=np.float32)
#     ones = np.ones((np.shape(P_xys)[0], 1))
#     P_xyzs = np.transpose(np.concatenate([P_xys/-focal_length, ones], axis=1)) # add -focal_length as new dim to (3 dim, n points)
#     M = get_M(opk)
#     vecs = np.transpose(np.matmul(np.transpose(M), P_xyzs))
#     return M, vecs

def cal_dist(vec1, spt1, vec2, spt2):
    spt1, spt2 = -spt1, -spt2
    p1, q1, r1 = spt1[:, 0], spt1[:, 1], spt1[:, 2]
    a1, b1, c1 = vec1[:, 0], vec1[:, 1], vec1[:, 2]
    p2, q2, r2 = spt2[:, 0], spt2[:, 1], spt2[:, 2]
    a2, b2, c2 = vec2[:, 0], vec2[:, 1], vec2[:, 2]
    a3 = a1*a2+b1*b2+c1*c2
    b3 = -(a1*a1+b1*b1+c1*c1)
    c3 = a1*(p2-p1)+b1*(q2-q1)+c1*(r2-r1)
    a4 = a2*a2+b2*b2+c2*c2
    b4 = -a3
    c4 = a2 * (p2-p1)+b2*(q2-q1)+c2*(r2-r1)
    t = (c3 * b4 - c4 * b3) / (a3 * b4 - a4 * b3)
    s = (a3 * c4 - a4 * c3) / (a3 * b4 - a4 * b3)
    xp = a1 * s - p1
    yp = b1 * s - q1
    zp = c1 * s - r1
    xq = a2 * t - p2
    yq = b2 * t - q2
    zq = c2 * t - r2
    dists_ecu=((xp-xq)**2+(yp-yq)**2+(zp-zq)**2)**(1/2)
    dists_blk = np.abs(xp-xq)+np.abs(yp-yq)+np.abs(zp-zq)
    pxyz1 = np.transpose(np.stack([xp, yp, zp]))
    pxyz2 = np.transpose(np.stack([xq, yq, zq]))
    return pxyz1, pxyz2, dists_ecu, dists_blk

def get_PQ(aerotri_params1, aerotri_params2, kp1_npidxs, kp2_npidxs, return_depth=False):
    """
    depth: in meter
    """
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET1, YOFFSET1 = aerotri_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2, XOFFSET2, YOFFSET2 = aerotri_params2
    M1, vecs1 = get_line_vecs(kp1_npidxs, OPK1, L_XYZ1, ROWS1, COLS1, PIXEL_SIZE1, FOCAL_LENGTH1, XOFFSET1, YOFFSET1)
    spts1 = np.tile(np.expand_dims(L_XYZ1, axis=0), [np.shape(vecs1)[0], 1])
    M2, vecs2 = get_line_vecs(kp2_npidxs, OPK2, L_XYZ2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2, XOFFSET2, YOFFSET2)
    spts2 = np.tile(np.expand_dims(L_XYZ2, axis=0), [np.shape(vecs2)[0], 1])
    pxyz1, pxyz2, dists_ecu, dists_blk = cal_dist(vecs1, spts1, vecs2, spts2)
    P_npidxs, ks1 = project_XYZs_to_npidxs(pxyz1, aerotri_params1, return_k=True)
    P_npidxs, ks2 = project_XYZs_to_npidxs(pxyz2, aerotri_params2, return_k=True)
    if return_depth:
        ks1 = (1 / ks1) * (FOCAL_LENGTH1 / 1000)
        ks2 = (1 / ks1) * (FOCAL_LENGTH2 / 1000)
    return ks1, ks2, pxyz1, pxyz2, dists_ecu, dists_blk


if __name__ == '__main__':
    vec1_test = np.array([[ 3, -1,  2], [ 1, -2,  1], [ 1,  2, -2]])  # a, b, c => vector
    spt1_test = np.array([[ 4,  2,  1], [ 3, -4, -2], [ 2, -3,  3]])  # p, q, r => L_point
    vec2_test = np.array([[-7,  1, -1], [ 4, -1,  2], [-3,  4,  1]])
    spt2_test = np.array([[ 5,  3, -2], [-9,  2,  0], [-2,  2,  0]])
    pxyz1, pxyz2, dists_ecu, dists_blk = cal_dist(vec1_test, spt1_test, vec2_test, spt2_test) # array([0.        , 7.87400787, 3.        ])
    print(dists_ecu)

