import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """
    
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    pyramid = [np.array(input_image, dtype='f')]
    for i in range(level):
        pyramid.append(utils.down_sampling(pyramid[-1]))
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).    
    pyramid = []
    for i in range(len(gaussian_pyramid)-1):
        pyramid.append(utils.safe_subtract(gaussian_pyramid[i], utils.up_sampling(gaussian_pyramid[i+1])))
    pyramid.append(gaussian_pyramid[-1])
    return pyramid

def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    M = gaussian_pyramid(mask, level)
    
    G1 = gaussian_pyramid(image1, level)
    G2 = gaussian_pyramid(image2, level)
    
    L1 = laplacian_pyramid(G1)
    L2 = laplacian_pyramid(G2)
    
    L = [utils.safe_add((1-m/255)*l1, m/255*l2) for (m, l1, l2) in zip(M, L1, L2)]
    
    output_image = L[-1]
    for l in reversed(L[:-1]):
        output_image = utils.safe_add(l, utils.up_sampling(output_image))
    
    return output_image


if __name__ == '__main__':
    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3
    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    # plt.show()

    ret = gaussian_pyramid(flame, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        # plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            # plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        # plt.show()
