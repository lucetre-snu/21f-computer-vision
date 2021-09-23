import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for padding reflection")
            
    
    pad = (size[0]//2, size[1]//2)
    shape = input_image.shape
    L = np.array([[0, pad[0], pad[0]+shape[0]-1, pad[0]+shape[0]+pad[0]-1],
                 [0, pad[1], pad[1]+shape[1]-1, pad[1]+shape[1]+pad[1]-1]])

    output_image = np.zeros((shape[0]+size[0]-1, shape[1]+size[1]-1, shape[2]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            output_image[pad[0]+i][pad[1]+j] = input_image[i][j]

    for i in range(3):
        for j in range(3):
            if i is 1 and j is 1:
                continue
            for y in range(L[0,i], L[0,i+1]+1):
                for x in range(L[1,j], L[1,j+1]+1):
                    if i is 1:
                        output_image[y][x] = output_image[y][L[1,j//2+1]*2-x]
                    elif j is 1:
                        output_image[y][x] = output_image[L[0,i//2+1]*2-y][x]
                    else:
                        output_image[y][x] = output_image[L[0,i//2+1]*2-y][L[1,j//2+1]*2-x]
                        
    assert((output_image == np.pad(input_image, pad_width=((size[0]//2,), (size[1]//2,), (0,)), mode='reflect')).all())
    
    return output_image

def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """

    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    
    return input_image


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")

    return input_image


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X direction
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code
    return


if __name__ == '__main__':
    image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    #image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5,5)) / 25.
    sigmax, sigmay = 5, 5
    ret = reflect_padding(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        #plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        #plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        #plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        #plt.show()

