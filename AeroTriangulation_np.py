import numpy as np

def get_M_np(OPK):
    """Collinearity Equation"""
    omega, phi, kappa = OPK[0], OPK[1], OPK[2]
    cO, sO = np.cos(omega), np.sin(omega)
    cP, sP = np.cos(phi), np.sin(phi)
    cK, sK = np.cos(kappa), np.sin(kappa)
    M = np.stack([[  cP*cK, cO*sK+sO*sP*cK, sO*sK-cO*sP*cK],
                  [ -cP*sK, cO*cK-sO*sP*sK, sO*cK+cO*sP*sK],
                  [     sP,         -sO*cP,          cO*cP]])
    return M

def convert_idxs_to_imxys_np(idxs, rows=13824, cols=7680, pixel_size=0.012):
    """
    image space linear transform
    pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    idxs
    """
    col_zoomout, row_zoomout = pixel_size, -pixel_size
    idxs_trans = np.transpose(idxs)
    row_idxs, col_idxs = idxs_trans[0, :], idxs_trans[1, :]
    xs = (col_idxs - (cols/2)) * col_zoomout
    ys = (row_idxs - (rows/2)) * row_zoomout
    imxys = np.transpose(np.stack([xs, ys]))
    return imxys

def get_line_vecs_np(idxs, opk, lxyz, rows=13824, cols=7680, pixel_size=0.012, focal_length=120):
    P_xys = convert_idxs_to_imxys_np(idxs, rows=rows, cols=cols, pixel_size=pixel_size) # (n points, 2 dim)
    focal_lengths = np.ones((np.shape(P_xys)[0], 1)) * -focal_length
    P_xyzs = np.transpose(np.concatenate([P_xys, focal_lengths], axis=1)) # add -focal_length as new dim to (3 dim, n points)
    M = get_M_np(opk)
    vecs = np.transpose(np.matmul(np.transpose(M), P_xyzs))
    return M, vecs

def cal_dist_np(vec1, spt1, vec2, spt2):
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

def get_PQ_np(aerotri_params1, aerotri_params2, kp1_npidxs, kp2_npidxs):
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1 = aerotri_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2 = aerotri_params2
    M1, vecs1 = get_line_vecs_np(kp1_npidxs, OPK1, L_XYZ1, ROWS1, COLS1, PIXEL_SIZE1, FOCAL_LENGTH1)
    spts1 = np.tile(np.expand_dims(L_XYZ1, axis=0), [np.shape(vecs1)[0], 1])
    M2, vecs2 = get_line_vecs_np(kp2_npidxs, OPK2, L_XYZ2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
    spts2 = np.tile(np.expand_dims(L_XYZ2, axis=0), [np.shape(vecs2)[0], 1])
    return cal_dist_np(vecs1, spts1, vecs2, spts2) # pxyz1, pxyz2, dists_ecu, dists_blk



if __name__ == '__main__':
    vec1_test = np.array([[ 3, -1,  2], [ 1, -2,  1], [ 1,  2, -2]])  # a, b, c => vector
    spt1_test = np.array([[ 4,  2,  1], [ 3, -4, -2], [ 2, -3,  3]])  # p, q, r => L_point
    vec2_test = np.array([[-7,  1, -1], [ 4, -1,  2], [-3,  4,  1]])
    spt2_test = np.array([[ 5,  3, -2], [-9,  2,  0], [-2,  2,  0]])
    pxyz1, pxyz2, dists_ecu, dists_blk = cal_dist_np(vec1_test, spt1_test, vec2_test, spt2_test) # array([0.        , 7.87400787, 3.        ])
    print(dists_ecu)

