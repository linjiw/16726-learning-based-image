# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

# def gradient(image):


def toy_recon(image):

    # For x gradient, g(x) = f(x + 1, y ) - f(x, y), where f is the image function
    # im2gy = np.zeros_like(image)
    # # sx = ndimage.sobel(img,axis=0,mode='constant')
    # im2gy[1:-1] = image[2:] - image[1:-1]
    # plt.imshow(im2gy)
    # plt.title('y-gradient')
    # plt.show()

    # im2gx = np.zeros_like(image)
    # # sx = ndimage.sobel(img,axis=0,mode='constant')
    # im2gx[:,1:-1] = image[:,2:] - image[:,1:-1]
    # plt.imshow(im2gx)
    # plt.title('x-gradient')
    # plt.show()
    """
    The first step is to write the objective function as a set of least squares constraints 
    in the standard matrix form: (Av-b)^2. 
    Here, “A” is a sparse matrix,
    “v” are the variables to be solved, 
    and “b” is a known vector.
    It is helpful to keep a matrix “im2var” that maps each pixel to a variable number, 
    such as:
    imh, imw, nb = im.shape
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int)
    """
    imh, imw = image.shape
    print(f"image.shape {image.shape}")
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int) 
    # im2var.shape (imh, imw)
    # b.shape (imh, imw)
    # Objective 1
    # for i in np.arange(imh * imw):
    #     A[i , im2var[]]
    
    total_len = imh * imw
    A = np.zeros((total_len * 2 + 1, total_len ))
    b = np.zeros((total_len * 2 + 1))

    e = 0
    for y in range(imh):
        for x in range(imw - 1):
            A[e, im2var[y, x + 1]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = image[y, x + 1] - image[y, x]
            e +=1

    for x in range(imw):
        for y in range(imh - 1):
        # if y != imh - 1: 
            A[e, im2var[y + 1, x]] = 1
            A[e, im2var[y, x]] = -1
            b[e] = image[y + 1, x] - image[y, x]
            e +=1
    A[-1, im2var[0, 0]] = 1
    b[-1] = image[0,0]
    print(f"A.shape {A.shape}\nb.shape {b.shape}\n")
    A = csc_matrix(A)

    v = lsqr(A, b, show=False)[0]
    v = v.reshape((imh, imw))
    print(f"v.shape {v.shape}")

    






    return v

def get_surrounding(index):
	i, j = index
	return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

# def if_border(index):
#     for i,j in get_surrounding(index):
#         if 


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    # use mask a a reference tool, 1: mask area, 0: outside
    imh, imw, cn = fg.shape
    all_v = np.zeros((imh, imw, cn))
    for channel in range(3):
        im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int) 

        total_len = imh * imw
        A = np.zeros((total_len * 4, total_len ))
        b = np.zeros(total_len * 4 )

        e = 0
        
        for y in range(imh - 1):
            for x in range(imw - 1):
                
                for ii in get_surrounding([y,x]):
                    if mask[ii[0], ii[1], 0]:
                    # print(f"e {e}")
                        A[e, im2var[ii[0], ii[1]]] = -1
                        A[e, im2var[y, x]] = 1
                        # print(fg[ii[0], ii[1], channel] - fg[y, x, channel])
                        b[e] = fg[y, x, channel] - fg[ii[0], ii[1], channel]
                    else:
                        A[e, im2var[y, x]] = 1
                        b[e] = bg[ii[0], ii[1], channel]
                        
                    e +=1
    
        print(f"A.shape {A.shape}\nb.shape {b.shape}\n")
    
        A = csc_matrix(A)

        v = lsqr(A, b, show=False)[0]
        v = v.reshape((imh, imw))
        print(f"v.shape {v.shape}")
        all_v[:, :, channel] = v
    bg[np.where(mask == 255)] = all_v[np.where(mask == 255)]







    return bg
    return fg * mask + bg * (1 - mask)


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    return fg * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.
        image_hat = toy_recon(image)

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = poisson_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
