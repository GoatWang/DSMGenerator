import os
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")



def get_M(OPK):
    """Collinearity Equation"""
    omega, phi, kappa = OPK[0], OPK[1], OPK[2]
    cO, sO = tf.cos(omega), tf.sin(omega)
    cP, sP = tf.cos(phi), tf.sin(phi)
    cK, sK = tf.cos(kappa), tf.sin(kappa)
    M = tf.stack([[  cP*cK, cO*sK+sO*sP*cK, sO*sK-cO*sP*cK],
                  [ -cP*sK, cO*cK-sO*sP*sK, sO*cK+cO*sP*sK],
                  [     sP,         -sO*cP,          cO*cP]])
    return M

def convert_npidxs_to_imxys(npidxs, rows=13824, cols=7680, pixel_size=0.012, x_offset=0, y_offset=0):
    """
    image space linear transform
    pixel_size = sensor_width/image_cols = 92.16/7680 = sensor_height/image_rows = 165.8885/13824  # unit: mm/px
    npidxs
    """
    col_zoomout, row_zoomout = pixel_size, -pixel_size
    npidxs_trans = tf.transpose(npidxs)
    row_idxs, col_idxs = npidxs_trans[0, :], npidxs_trans[1, :]
    xs = (col_idxs - (cols/2 + x_offset/pixel_size)) * col_zoomout
    ys = (row_idxs - (rows/2 + y_offset/pixel_size)) * row_zoomout
    imxys = tf.transpose(tf.stack([xs, ys]))
    return imxys

def get_line_vecs(npidxs, opk, lxyz, rows=13824, cols=7680, pixel_size=0.012, focal_length=120, x_offset=0, y_offset=0):
    P_xys = convert_npidxs_to_imxys(npidxs, rows=rows, cols=cols, pixel_size=pixel_size, x_offset=x_offset, y_offset=y_offset) # (n points, 2 dim)
    focal_lengths = tf.ones((tf.shape(P_xys)[0], 1)) * -focal_length
    P_xyzs = tf.transpose(tf.concat([P_xys, focal_lengths], axis=1)) # add -focal_length as new dim to (3 dim, n points)
    M = get_M(opk)
    vecs = tf.transpose(tf.matmul(tf.transpose(M), P_xyzs))
    return  M, vecs

def cal_dist(vec1, spt1, vec2, spt2):
    _spt1, _spt2 = -spt1, -spt2
    p1, q1, r1 = _spt1[:, 0], _spt1[:, 1], _spt1[:, 2]
    a1, b1, c1 = vec1[:, 0], vec1[:, 1], vec1[:, 2]
    p2, q2, r2 = _spt2[:, 0], _spt2[:, 1], _spt2[:, 2]
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
    _dist_ecu = ((xp-xq)**2+(yp-yq)**2+(zp-zq)**2)**(1/2)
    _dist_blk = tf.abs(xp-xq)+tf.abs(yp-yq)+tf.abs(zp-zq)
    _pxyz1 = tf.transpose(tf.stack([xp, yp, zp]))
    _pxyz2 = tf.transpose(tf.stack([xq, yq, zq]))
    return _pxyz1, _pxyz2, _dist_ecu, _dist_blk

def train(loss, learning_rate):
#     return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)

# def optimize_opk(OPK1, OPK2, L_XYZ1, L_XYZ2, kp1_npidxs, kp2_npidxs, learning_rate=0.0001, training_steps=500, verbose=10, seed=2020):
def get_PQ(aerotri_params1, aerotri_params2, kp1_npidxs, kp2_npidxs):
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET, YOFFSET = aerotri_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2, XOFFSET, YOFFSET = aerotri_params2

    L_st = np.array([1, 1, L_XYZ1[2]], dtype=np.float32)
    L_XYZ1_temp = (L_XYZ1 - L_XYZ1) + L_st
    L_XYZ2_temp = (L_XYZ2 - L_XYZ1) + L_st
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        # input param
        npidxs1 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs1')
        npidxs2 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs2')
        feed_dict = {npidxs1:kp1_npidxs, npidxs2:kp2_npidxs}

        opk1 = tf.Variable(list(OPK1), shape=3, trainable=True, name="opk1")
        opk2 = tf.Variable(list(OPK2), shape=3, trainable=True, name="opk2")
        lxyz1 = tf.Variable(list(L_XYZ1_temp), shape=3, trainable=False, name="lxyz1")
        lxyz2 = tf.Variable(list(L_XYZ2_temp), shape=3, trainable=True, name="lxyz2")

        # tf operations
        M1, vecs1 = get_line_vecs(npidxs1, opk1, lxyz1, ROWS1, COLS1, PIXEL_SIZE1, FOCAL_LENGTH1)
        spts1 = tf.tile(tf.expand_dims(lxyz1, axis=0), [tf.shape(vecs1)[0], 1])
        M2, vecs2 = get_line_vecs(npidxs2, opk2, lxyz2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
        spts2 = tf.tile(tf.expand_dims(lxyz2, axis=0), [tf.shape(vecs2)[0], 1])
        pxyz1, pxyz2, dists_ecu, dists_blk = cal_dist(vecs1, spts1, vecs2, spts2)
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        pxyz1_out, pxyz2_out, dists_ecu_out = sess.run([pxyz1, pxyz2, dists_ecu], feed_dict=feed_dict)

    pxyz1_out = L_XYZ1 + (pxyz1_out - L_st)
    pxyz2_out = L_XYZ1 + (pxyz2_out - L_st)

    return pxyz1_out, pxyz2_out, dists_ecu_out


def optimize_opk(aerotri_params1, aerotri_params2, kp1_npidxs, kp2_npidxs, OPK1_trainable=False, L_XYZ1_trainable=False, OPK2_trainable=True, L_XYZ2_trainable=True, std_lim_for_training=None, learning_rate=0.0001, training_steps=500, verbose=10, seed=2020):
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET, YOFFSET = aerotri_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2, XOFFSET, YOFFSET = aerotri_params2
    OPK1, L_XYZ1, OPK2, L_XYZ2 = np.array(OPK1, dtype=np.float32), np.array(L_XYZ1, dtype=np.float32), np.array(OPK2, dtype=np.float32), np.array(L_XYZ2, dtype=np.float32)

    L_st = np.array([1, 1, L_XYZ1[2]], dtype=np.float32)
    L_XYZ1_temp = (L_XYZ1 - L_XYZ1) + L_st
    L_XYZ2_temp = (L_XYZ2 - L_XYZ1) + L_st
    
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)
    with tf.Session() as sess:
        # input & variable
        npidxs1 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs1')
        npidxs2 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs2')
        feed_dict = {npidxs1:kp1_npidxs, npidxs2:kp2_npidxs}


        opk1 = tf.Variable(list(OPK1), shape=3, trainable=OPK1_trainable, name="opk1")
        opk2 = tf.Variable(list(OPK2), shape=3, trainable=OPK2_trainable, name="opk2")
        lxyz1 = tf.Variable(list(L_XYZ1_temp), shape=3, trainable=L_XYZ1_trainable, name="lxyz1")
        lxyz2 = tf.Variable(list(L_XYZ2_temp), shape=3, trainable=L_XYZ2_trainable, name="lxyz2")
        

        # tf operations
        M1, vecs1 = get_line_vecs(npidxs1, opk1, lxyz1, ROWS1, COLS1, PIXEL_SIZE1, FOCAL_LENGTH1)
        spts1 = tf.tile(tf.expand_dims(lxyz1, axis=0), [tf.shape(vecs1)[0], 1])
        M2, vecs2 = get_line_vecs(npidxs2, opk2, lxyz2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
        spts2 = tf.tile(tf.expand_dims(lxyz2, axis=0), [tf.shape(vecs2)[0], 1])
        pxyz1, pxyz2, dists_ecu, dists_blk = cal_dist(vecs1, spts1, vecs2, spts2)
        dists_ecu_mean = tf.reduce_mean(dists_ecu)
        dists_blk_mean = tf.reduce_mean(dists_blk)
        l_dist = tf.sqrt(tf.reduce_sum(tf.square(lxyz1 - lxyz2)))
        loss_ecu = dists_ecu_mean + 1/l_dist
        loss_blk = dists_blk_mean + 1/l_dist
        train_ecu_op = train(dists_ecu_mean, learning_rate)
        train_blk_op = train(dists_blk_mean, learning_rate)
        train_ecu_ldist_op = train(loss_ecu, learning_rate)
        train_blk_ldist_op = train(loss_blk, learning_rate)
#         ddist_dopk = tf.gradients(dists_ecu_mean, opk1)

        # run operation
        sess.run(tf.global_variables_initializer())

        # update feed dict for precise training
        if std_lim_for_training:
            dist_ecu_out = sess.run(dists_ecu, feed_dict=feed_dict)
            # training_idxs = np.argsort(dist_ecu_out)[:top_n_for_training]
            training_idxs = np.where(dist_ecu_out < (np.std(dist_ecu_out) * std_lim_for_training))[0]
            assert len(training_idxs) > 0, "You don't have tie point pairs with distance less than std_lim_for_training*std"
            feed_dict = {npidxs1:kp1_npidxs[training_idxs], npidxs2:kp2_npidxs[training_idxs]}
        else:
            training_idxs = np.arange(0, len(kp1_npidxs), 1)

        print("distance (init):", sess.run(dists_ecu_mean, feed_dict=feed_dict))
        non_zero_idx = None
        l_dist_out = 0
        for step in range(training_steps):
            if non_zero_idx is None:
                if l_dist_out > 0.1:
                    sess.run(train_ecu_op, feed_dict=feed_dict)
                else:
                    sess.run(train_ecu_ldist_op, feed_dict=feed_dict)
            else:
                if l_dist_out > 0.1:
                    sess.run(train_blk_op, feed_dict=feed_dict)
                else:
                    sess.run(train_blk_ldist_op, feed_dict=feed_dict)
                

            dists_ecu_out, dists_blk_out, dists_ecu_mean_out, l_dist_out = sess.run([dists_ecu, dists_blk, dists_ecu_mean, l_dist], feed_dict=feed_dict)
            if np.min(dists_ecu_out) == 0:
                zero_idx = np.where(dists_ecu_out == 0)[0]
                non_zero_idx = list(set(range(len(dists_ecu_out))) - set(zero_idx))
            else:
                non_zero_idx = None

            if verbose != 0:
                if step % verbose == 0:
                    print(step, 'mean distance:', np.mean(dists_ecu_out))
                    
        opk1_out, opk2_out, lxyz1_out, lxyz2_out, pxyz1_out, pxyz2_out = sess.run([opk1, opk2, lxyz1, lxyz2, pxyz1, pxyz2], feed_dict=feed_dict)

    lxyz1_out = L_XYZ1 + (lxyz1_out - L_st) # move the init point from L_st to L_XYZ1
    lxyz2_out = L_XYZ1 + (lxyz2_out - L_st)
    pxyz1_out = L_XYZ1 + (pxyz1_out - L_st)
    pxyz2_out = L_XYZ1 + (pxyz2_out - L_st)
    return training_idxs, opk1_out, opk2_out, lxyz1_out, lxyz2_out, pxyz1_out, pxyz2_out



def optimize_opk_3imgs(aerotri_params1, aerotri_params2, aerotri_params3, kp1_npidxs, kp2_npidxs, kp3_npidxs, OPK1_trainable=False, L_XYZ1_trainable=False, OPK2_trainable=True, L_XYZ2_trainable=True, OPK3_trainable=True, L_XYZ3_trainable=True, learning_rate=0.0001, training_steps=500, verbose=10, seed=2020):
    OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET, YOFFSET = aerotri_params1
    OPK2, L_XYZ2, ROWS2, COLS2, FOCAL_LENGTH2, PIXEL_SIZE2, XOFFSET, YOFFSET = aerotri_params2
    OPK3, L_XYZ3, ROWS3, COLS3, FOCAL_LENGTH3, PIXEL_SIZE3, XOFFSET, YOFFSET = aerotri_params3
    OPK1, L_XYZ1, OPK2, L_XYZ2, OPK3, L_XYZ3 = np.array(OPK1, dtype=np.float32), np.array(L_XYZ1, dtype=np.float32), np.array(OPK2, dtype=np.float32), np.array(L_XYZ2, dtype=np.float32), np.array(OPK3, dtype=np.float32), np.array(L_XYZ3, dtype=np.float32)

    L_st = np.array([1, 1, 1], dtype=np.float32)
    L_XYZ1_temp = (L_XYZ1 - L_XYZ1) + L_st
    L_XYZ2_temp = (L_XYZ2 - L_XYZ1) + L_st
    L_XYZ3_temp = (L_XYZ3 - L_XYZ1) + L_st
    
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)
    with tf.Session() as sess:
        # input & variable
        npidxs121 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs121')
        npidxs122 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs122')
        npidxs232 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs232')
        npidxs233 = tf.placeholder(tf.float32, shape=(None, 2), name='npidxs233')
        feed_dict = {npidxs121:kp1_npidxs, npidxs122:kp2_npidxs, npidxs232:kp2_npidxs, npidxs233:kp3_npidxs}
        
        opk1 = tf.Variable(list(OPK1), shape=3, trainable=OPK1_trainable, name="opk1")
        opk2 = tf.Variable(list(OPK2), shape=3, trainable=OPK2_trainable, name="opk2")
        opk3 = tf.Variable(list(OPK3), shape=3, trainable=OPK3_trainable, name="opk3")
        lxyz1 = tf.Variable(list(L_XYZ1_temp), shape=3, trainable=L_XYZ1_trainable, name="lxyz1")
        lxyz2 = tf.Variable(list(L_XYZ2_temp), shape=3, trainable=L_XYZ2_trainable, name="lxyz2")
        lxyz3 = tf.Variable(list(L_XYZ3_temp), shape=3, trainable=L_XYZ3_trainable, name="lxyz3")
        
        # tf operations
        M121, vecs121 = get_line_vecs(npidxs121, opk1, lxyz1, ROWS1, COLS1, PIXEL_SIZE1, FOCAL_LENGTH1)
        spts121 = tf.tile(tf.expand_dims(lxyz1, axis=0), [tf.shape(vecs121)[0], 1])
        M122, vecs122 = get_line_vecs(npidxs122, opk2, lxyz2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
        spts122 = tf.tile(tf.expand_dims(lxyz2, axis=0), [tf.shape(vecs122)[0], 1])
        M232, vecs232 = get_line_vecs(npidxs232, opk2, lxyz2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
        spts232 = tf.tile(tf.expand_dims(lxyz2, axis=0), [tf.shape(vecs232)[0], 1])
        M233, vecs233 = get_line_vecs(npidxs233, opk3, lxyz3, ROWS3, COLS3, PIXEL_SIZE3, FOCAL_LENGTH3)
        spts233 = tf.tile(tf.expand_dims(lxyz3, axis=0), [tf.shape(vecs233)[0], 1])

        pxyz121, pxyz122, dists_ecu12, dists_blk12 = cal_dist(vecs121, spts121, vecs122, spts122)
        pxyz232, pxyz233, dists_ecu23, dists_blk23 = cal_dist(vecs122, spts122, vecs233, spts233)

        dists_ecu_mean12, dists_blk_mean12 = tf.reduce_mean(dists_ecu12), tf.reduce_mean(dists_blk12)
        dists_ecu_mean23, dists_blk_mean23 = tf.reduce_mean(dists_ecu23), tf.reduce_mean(dists_blk23)
        overlappoint_dist_ecu = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(pxyz122 - pxyz232), axis=1)))
        overlappoint_dist_blk = tf.reduce_mean(tf.abs(pxyz122 - pxyz232))
        train_ecu_op = train(dists_ecu_mean12 + dists_ecu_mean23 + overlappoint_dist_ecu, learning_rate)
        train_blk_op = train(dists_blk_mean12 + dists_blk_mean23 + overlappoint_dist_blk, learning_rate)

        # run operation
        sess.run(tf.global_variables_initializer())

        # update feed dict for precise training
        training_idxs = np.arange(0, len(kp1_npidxs), 1)

        print("distance 12 (init):", sess.run(dists_ecu_mean12, feed_dict=feed_dict))
        print("distance 23 (init):", sess.run(dists_ecu_mean23, feed_dict=feed_dict))
        non_zero_idx = None
        for step in range(training_steps):
            if non_zero_idx is None:
                sess.run(train_ecu_op, feed_dict=feed_dict)
            else:
                sess.run(train_blk_op, feed_dict=feed_dict)

            dists_ecu12_out, dists_ecu23_out, overlappoint_dist_ecu_out = sess.run([dists_ecu12, dists_ecu23, overlappoint_dist_ecu], feed_dict=feed_dict)
            non_zero_idx = True if (np.min(dists_ecu12_out) == 0) or (np.min(dists_ecu23_out) == 0) or (np.min(overlappoint_dist_ecu_out) == 0) or (np.std(overlappoint_dist_ecu_out) == 0) else None

            if verbose != 0:
                if step % verbose == 0:
                    print(step, 'mean distance 12:', np.mean(dists_ecu12_out))
                    print(step, 'mean distance 23:', np.mean(dists_ecu23_out))

        opk1_out, opk2_out, opk3_out, lxyz1_out, lxyz2_out, lxyz3_out, pxyz1_out, pxyz2_out, pxyz3_out = sess.run([opk1, opk2, opk3, lxyz1, lxyz2, lxyz3, pxyz121, pxyz122, pxyz233], feed_dict=feed_dict)

    lxyz1_out = L_XYZ1 + (lxyz1_out - L_st) # move the init point from L_st to L_XYZ1
    lxyz2_out = L_XYZ1 + (lxyz2_out - L_st)
    lxyz3_out = L_XYZ1 + (lxyz3_out - L_st)
    pxyz1_out = L_XYZ1 + (pxyz1_out - L_st)
    pxyz2_out = L_XYZ1 + (pxyz2_out - L_st)
    pxyz3_out = L_XYZ1 + (pxyz3_out - L_st)
    return training_idxs, opk1_out, opk2_out, opk3_out, lxyz1_out, lxyz2_out, lxyz3_out, pxyz1_out, pxyz2_out, pxyz3_out


def optimize_opk_3imgs(aerotri_params_imgs, kp_npidxs_imgs, OPK_trainable_imgs, L_XYZ_trainable_imgs, learning_rate=0.0001, training_steps=500, verbose=10, seed=2020):
    L_st = np.array([1, 1, 1], dtype=np.float32)
    L_XYZ1 = None
    ap_dicts = []
    for aerotri_params, OPK_trainable, L_XYZ_trainable in zip(aerotri_params_imgs, OPK_trainable_imgs, L_XYZ_trainable_imgs):
        OPK, L_XYZ, ROWS, COLS, FOCAL_LENGTH, PIXEL_SIZE, XOFFSET, YOFFSET = aerotri_params
        L_XYZ1 = L_XYZ if L_XYZ1 is None else L_XYZ1
        ap_dicts.append({
            'OPK': OPK, 
            'OPK_trainable': OPK_trainable,
            'L_XYZ_temp': (L_XYZ - L_XYZ1) + L_st, 
            'L_XYZ_trainable': L_XYZ_trainable, 
            'ROWS': ROWS, 
            'COLS': COLS, 
            'FOCAL_LENGTH': FOCAL_LENGTH, 
            'PIXEL_SIZE': PIXEL_SIZE})
    
    tf.reset_default_graph()
    tf.random.set_random_seed(seed)
    with tf.Session() as sess:
        # input & variable
        # kp_npidxs_dict_pl = {i: tf.placeholder(tf.float32, shape=(None, 2)) for i in range(len(kp_npidxs_imgs))}
        # feed_dict = {kp_npidxs_dict_pl[i]:kp_npidxs for i, kp_npidxs in enumerate(kp_npidxs_imgs)}
        kp_npidxs = [tf.Variable(kp_npidxs, shape=3, trainable=False, name="kp_npidxs" + str(i)) for i, kp_npidxs in enumerate(kp_npidxs_imgs)]
        opks = [tf.Variable(list(ap_dict['OPK']), shape=3, trainable=ap_dict['OPK_trainable'], name="opk" + str(i)) for i, ap_dict in enumerate(ap_dicts)]
        lxyzs = [tf.Variable(list(ap_dict['L_XYZ1_temp']), shape=3, trainable=ap_dict['L_XYZ_trainable'], name="lxyzs" + str(i)) for i, ap_dict in enumerate(ap_dicts)]
        
        # tf operations
        for kp_npidx, opk, lxyz, ap_dict in zip(kp_npidxs, opks, lxyzs, ap_dicts):
            M, vecs = get_line_vecs(kp_npidx, opk, lxyz, ap_dict['ROWS'], ap_dict['COLS'], ap_dict['PIXEL_SIZE'], ap_dict['FOCAL_LENGTH'])
            spts = tf.tile(tf.expand_dims(lxyz, axis=0), [tf.shape(vecs)[0], 1])




        M121, vecs121 = get_line_vecs(npidxs121, opk1, lxyz1, ROWS1, COLS1, PIXEL_SIZE1, FOCAL_LENGTH1)
        spts121 = tf.tile(tf.expand_dims(lxyz1, axis=0), [tf.shape(vecs121)[0], 1])
        M122, vecs122 = get_line_vecs(npidxs122, opk2, lxyz2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
        spts122 = tf.tile(tf.expand_dims(lxyz2, axis=0), [tf.shape(vecs122)[0], 1])
        M232, vecs232 = get_line_vecs(npidxs232, opk2, lxyz2, ROWS2, COLS2, PIXEL_SIZE2, FOCAL_LENGTH2)
        spts232 = tf.tile(tf.expand_dims(lxyz2, axis=0), [tf.shape(vecs232)[0], 1])
        M233, vecs233 = get_line_vecs(npidxs233, opk3, lxyz3, ROWS3, COLS3, PIXEL_SIZE3, FOCAL_LENGTH3)
        spts233 = tf.tile(tf.expand_dims(lxyz3, axis=0), [tf.shape(vecs233)[0], 1])

        pxyz121, pxyz122, dists_ecu12, dists_blk12 = cal_dist(vecs121, spts121, vecs122, spts122)
        pxyz232, pxyz233, dists_ecu23, dists_blk23 = cal_dist(vecs122, spts122, vecs233, spts233)

        dists_ecu_mean12, dists_blk_mean12 = tf.reduce_mean(dists_ecu12), tf.reduce_mean(dists_blk12)
        dists_ecu_mean23, dists_blk_mean23 = tf.reduce_mean(dists_ecu23), tf.reduce_mean(dists_blk23)
        overlappoint_dist_ecu = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(pxyz122 - pxyz232), axis=1)))
        overlappoint_dist_blk = tf.reduce_mean(tf.abs(pxyz122 - pxyz232))
        train_ecu_op = train(dists_ecu_mean12 + dists_ecu_mean23 + overlappoint_dist_ecu, learning_rate)
        train_blk_op = train(dists_blk_mean12 + dists_blk_mean23 + overlappoint_dist_blk, learning_rate)

        # run operation
        sess.run(tf.global_variables_initializer())

        # update feed dict for precise training
        training_idxs = np.arange(0, len(kp1_npidxs), 1)

        print("distance 12 (init):", sess.run(dists_ecu_mean12, feed_dict=feed_dict))
        print("distance 23 (init):", sess.run(dists_ecu_mean23, feed_dict=feed_dict))
        non_zero_idx = None
        for step in range(training_steps):
            if non_zero_idx is None:
                sess.run(train_ecu_op, feed_dict=feed_dict)
            else:
                sess.run(train_blk_op, feed_dict=feed_dict)

            dists_ecu12_out, dists_ecu23_out, overlappoint_dist_ecu_out = sess.run([dists_ecu12, dists_ecu23, overlappoint_dist_ecu], feed_dict=feed_dict)
            non_zero_idx = True if (np.min(dists_ecu12_out) == 0) or (np.min(dists_ecu23_out) == 0) or (np.min(overlappoint_dist_ecu_out) == 0) or (np.std(overlappoint_dist_ecu_out) == 0) else None

            if verbose != 0:
                if step % verbose == 0:
                    print(step, 'mean distance 12:', np.mean(dists_ecu12_out))
                    print(step, 'mean distance 23:', np.mean(dists_ecu23_out))

        opk1_out, opk2_out, opk3_out, lxyz1_out, lxyz2_out, lxyz3_out, pxyz1_out, pxyz2_out, pxyz3_out = sess.run([opk1, opk2, opk3, lxyz1, lxyz2, lxyz3, pxyz121, pxyz122, pxyz233], feed_dict=feed_dict)

    lxyz1_out = L_XYZ1 + (lxyz1_out - L_st) # move the init point from L_st to L_XYZ1
    lxyz2_out = L_XYZ1 + (lxyz2_out - L_st)
    lxyz3_out = L_XYZ1 + (lxyz3_out - L_st)
    pxyz1_out = L_XYZ1 + (pxyz1_out - L_st)
    pxyz2_out = L_XYZ1 + (pxyz2_out - L_st)
    pxyz3_out = L_XYZ1 + (pxyz3_out - L_st)
    return training_idxs, opk1_out, opk2_out, opk3_out, lxyz1_out, lxyz2_out, lxyz3_out, pxyz1_out, pxyz2_out, pxyz3_out








# if __name__ == '__main__':
#     tf.reset_default_graph()
#     vec1_test = np.array([[3, -1, 2], [1, -2, 1], [1, 2, -2]])  # p, q, r => L_point
#     spt1_test = np.array([[4, 2, 1], [3, -4, -2], [2, -3, 3]])  # a, b, c => vector
#     vec2_test = np.array([[-7, 1, -1], [4, -1, 2], [-3, 4, 1]])
#     spt2_test = np.array([[5, 3, -2], [-9, 2, 0], [-2, 2, 0]])
    
#     with tf.Session() as sess:
#         # input param
#         vecs1 = tf.placeholder(tf.float32, shape=(None, 3))
#         spts1 = tf.placeholder(tf.float32, shape=(None, 3))
#         vecs2 = tf.placeholder(tf.float32, shape=(None, 3))
#         spts2 = tf.placeholder(tf.float32, shape=(None, 3))
#         feed_dict = {vecs1:vec1_test, spts1:spt1_test, vecs2:vec2_test, spts2:spt2_test}

#         # tf operations
#         sess.run(tf.global_variables_initializer())
#         pxyz1, pxyz2, dist_ecu, dist_blk = sess.run(cal_dist(vecs1, spts1, vecs2, spts2), feed_dict=feed_dict)
#     print(dist_ecu) # array([0.       , 7.8740077, 3.       ], dtype=float32)


if __name__ == '__main__':
    import os 
    import cv2
    import TronGisPy as tgp
    from datetime import datetime
    from matplotlib import pyplot as plt
    from io_aereo_params import get_DMC_aereo_params
    from TiePoints import get_PQ as get_PQ_np
    from TiePoints import find_tie_points_grids, filter_tie_points_by_PQ_dist, plot_kp_lines, plot_aero_triangulation

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

    # find tie points
    kp1_pts, kp2_pts = find_tie_points_grids(gray1, gray2, nfeatures=1000, topn_n_matches=300, grids=(3, 3)) # find_tie_points_farest(gray1, gray2)
    dep1, dep2, pxyz1, pxyz2, kp1_pts, kp2_pts, dists_ecu = filter_tie_points_by_PQ_dist(aereo_params1, aereo_params2, kp1_pts, kp2_pts, max_dist=3)
    texts_left = ["d=%.2f, "%d + "k=%.2f"%k for idx, (d, k) in enumerate(zip(dists_ecu, dep1))]
    # plot_kp_lines(img1_norm, kp1_pts, img2_norm, kp2_pts, texts_left=texts_left, fontsize=12)

    kp1_paired_pts_npidxs = np.array([kp1_pts[:, 1], kp1_pts[:, 0]]).T
    kp2_paired_pts_npidxs = np.array([kp2_pts[:, 1], kp2_pts[:, 0]]).T
    opt_outs = optimize_opk(aereo_params1, aereo_params2,
                            kp1_paired_pts_npidxs, kp2_paired_pts_npidxs, 
                            OPK1_trainable=False, L_XYZ1_trainable=False, 
                            OPK2_trainable=True, L_XYZ2_trainable=False, 
                            std_lim_for_training = 0.8,
                            learning_rate=0.0001, training_steps=100, 
                            verbose=10, seed=2020)

    training_idxs, OPK1_new, OPK2_new, L_XYZ1_new, L_XYZ2_new, P_XYZ1_new, P_XYZ2_new = opt_outs
    aereo_params1_new = np.array(aereo_params1).copy() #OPK1, L_XYZ1, ROWS1, COLS1, FOCAL_LENGTH1, PIXEL_SIZE1, XOFFSET1, YOFFSET1
    aereo_params1_new[0] = OPK1_new
    aereo_params1_new[1] = L_XYZ1_new
    aereo_params2_new = np.array(aereo_params2).copy()
    aereo_params2_new[0] = OPK2_new
    aereo_params2_new[1] = L_XYZ2_new

    # plot calibration result
    P_XYZ1, P_XYZ2, dists_ecu = get_PQ(aereo_params1, aereo_params2, kp1_paired_pts_npidxs[training_idxs], kp2_paired_pts_npidxs[training_idxs])
    plt.figure(figsize=(10, 5))
    plt.hist(np.sqrt(np.sum(np.square(P_XYZ1 - P_XYZ2), axis=1)), color=(1, 0, 0, 0.5), bins=30, label='original')
    plt.hist(np.sqrt(np.sum(np.square(P_XYZ1_new - P_XYZ2_new), axis=1)), color=(0, 0, 1, 0.5), bins=30, label='calibrated')
    plt.legend()
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("plot calibration result 3d")
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title('Before Calibration')
    plot_aero_triangulation(aereo_params1, aereo_params2, P_XYZ1, P_XYZ2, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title('After Calibration')
    plot_aero_triangulation(aereo_params1_new, aereo_params2_new, P_XYZ1_new, P_XYZ2_new, ax=ax2)
    plt.tight_layout()
    # plt.show()


    dt_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    d = os.path.join('temp', '3d', dt_str)
    os.mkdir(d)
    for angle in np.arange(0, 360, 5): # rotate the axes and update
        ax1.view_init(0, angle)
        ax2.view_init(0, angle)
        plt.draw()
        # plt.pause(.001)
        plt.savefig(os.path.join(d, str(angle).zfill(3)+".png"))

