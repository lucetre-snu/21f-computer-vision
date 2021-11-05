import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    # SVD of A such that p1 = A p2
    assert(p1.shape == p2.shape)
    N = p1.shape[0]
    A = np.zeros((2*N, 9))
    for i in range(N):
        x1, y1 = p1[i]
        x2, y2 = p2[i]
        A[i*2:i*2+2,:] = np.array([[x2, y2, 1, 0, 0, 0, -x2*x1, -y2*x1, -x1],
                                   [0, 0, 0, x2, y2, 1, -x2*y1, -y2*y1, -y1]])

    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H

def compute_h_norm(p1, p2):
    # Nomalize H
    y_max, x_max = np.max([np.max(p1, axis=0), np.max(p2, axis=0)], axis=0)
    H = compute_h(p1 / (y_max, x_max), p2 / (y_max, x_max))
    
    H = H / np.sqrt(np.sum(H**2))
    H[0,:] *= y_max
    H[1,:] *= x_max
    H[:,0] /= y_max
    H[:,1] /= x_max
    
    return H / H[-1,-1]

def warp_image(igs_in, igs_ref, H):
    # Backward warping & merge images 
    height, width, _ = igs_in.shape   
    offset = height//2
    p_ref = []
    for x in range(-width, width):
        for y in range(2*height):
            point = [x, y-offset, 1]
            p_ref.append(point)
    p_ref = np.array(p_ref).transpose()
    
    H_inv = np.linalg.pinv(H)

    p_in = np.matmul(H_inv, p_ref)
    p_in = p_in / p_in[2, :]
    p_in = p_in[:2, :]
    p_in = np.round(p_in, 0).astype(np.int)

    igs_warp = np.zeros((height, width, 3), dtype = np.uint8)
    igs_merge = np.zeros((height*2, width*2, 3), dtype = np.uint8)
    for pt1, pt2 in zip(p_ref[:2, :].transpose(), p_in.transpose()):        
        if 0 <= pt2[1] < height and 0 <= pt2[0] < width:
            if 0 <= pt1[1] < height and 0 <= pt1[0] < width:
                igs_warp[pt1[1], pt1[0]] = igs_in[pt2[1], pt2[0]]
            igs_merge[pt1[1]+offset, pt1[0]+width] = igs_in[pt2[1], pt2[0]]
    igs_merge[offset:offset+height, width:width*2] = igs_ref
    
    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    # Rectify
    H = compute_h_norm(p2, p1)
    igs_rec, _ = warp_image(igs, igs, H)

    return igs_rec
  
def set_cor_mosaic():
    # Set corresponding points for mosaic
    p_in = np.array([[958, 417],
                    [961, 821],
                    [1239, 541],
                    [1254, 959],
                    [1327, 909],
                    [1326, 580],
                    [1445, 504],
                    [1440, 405],
                    [1281, 416],
                    [1283, 509],
                    [1178, 869],
                    [1429, 895],
                    [1581, 908],
                    [1498, 1140],
                    [1487, 1174],
                    [833, 382],
                    [834, 817],
                    [1219, 183],
                    [1287, 253],
                    [1075, 72]])
    
    p_ref = np.array([[199, 403],
                    [202, 839],
                    [493, 542],
                    [509, 947],
                    [576, 893],
                    [577, 583],
                    [679, 513],
                    [677, 422],
                    [535, 424],
                    [536, 513],
                    [435, 868],
                    [665, 870],
                    [786, 869],
                    [720, 1084],
                    [713, 1115],
                    [43, 351],
                    [46, 848],
                    [476, 191],
                    [539, 267],
                    [329, 54]])
    
    return p_in, p_ref

def set_cor_rec():
    # Set corresponding points for rectifying
    c_in = np.array([[1381, 163],
                    [1381, 831],
                    [1061, 820],
                    [1069, 194]])
    
    c_ref = np.array([[1200, 50],
                    [1200, 1000],
                    [700, 1000],
                    [700, 50]])
                      
    return c_in, c_ref


def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_mergeed.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
