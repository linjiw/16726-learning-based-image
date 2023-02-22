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
    
    top = None
    bot = None
    left = None
    right = None
    pad = 5
    for y in range(imh):
        if np.sum(mask[y]) > 0:
            top = y - pad
            break
    for y in reversed(range(imh)):
        if np.sum(mask[y]) > 0:
            bot = y + pad
            break
    for y in range(imw):
        if np.sum(mask[:,y]) > 0:
            left = y - pad
            break
    for y in reversed(range(imw)):
        if np.sum(mask[:,y]) > 0:
            right = y + pad
            break
    print(f"top {top}\nbot {bot}\nleft {left}\nright {right}\n")
    # test_img = np.zeros((imh, imw))
    # test_img[top:bot,left:right] = 100
    # print(test_img)
    # print(f"sum {np.sum(test_img)}")
    # plt.imshow(test_img)
    # plt.savefig(f"test bbox.png")
    
    mask_width = np.abs(right - left)
    mask_height = np.abs(bot - top)

    all_v = np.zeros((mask_height, mask_width, cn))

# if abs(s_i - s_j) >= abs (t_i - t_j):
#     d_ij = s_i - s_j
# else:
#     d_ij = t_i - t_j

    for channel in range(3):
        im2var = np.arange(mask_width * mask_height).reshape((mask_height, mask_width)).astype(int) 

        total_len = mask_width * mask_height
        A = np.zeros((total_len * 4, total_len ))
        b = np.zeros(total_len * 4 )
        e = 0
        for y in range(mask_height):
            for x in range(mask_width):
                
                for ii in get_surrounding([y,x]):
                    if mask[top + ii[0], left + ii[1], 0]:
                    # print(f"e {e}")
                        A[e, im2var[ii[0], ii[1]]] = -1
                        A[e, im2var[y, x]] = 1
                        # print(fg[ii[0], ii[1], channel] - fg[y, x, channel])
                        # if abs(fg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel]) >= abs(bg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel])
                        b[e] = fg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel]


                    else:
                        A[e, im2var[y, x]] = 1
                        b[e] = bg[top + ii[0], left + ii[1], channel]
                        
                    e +=1
    
        print(f"A.shape {A.shape}\nb.shape {b.shape}\n")
    
        A = csc_matrix(A)

        v = lsqr(A, b, show=False)[0]
        v = v.reshape((mask_height, mask_width))
        print(f"v.shape {v.shape}")
        all_v[:, :, channel] = v
    # bg[np.where(mask == 255)] = all_v[np.where(mask == 255)]
    ans = bg.copy()
    ans[top :bot, left :right] = all_v







    return ans
    return fg * mask + bg * (1 - mask)


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    imh, imw, cn = fg.shape
    
    top = None
    bot = None
    left = None
    right = None
    pad = 5
    for y in range(imh):
        if np.sum(mask[y]) > 0:
            top = y - pad
            break
    for y in reversed(range(imh)):
        if np.sum(mask[y]) > 0:
            bot = y + pad
            break
    for y in range(imw):
        if np.sum(mask[:,y]) > 0:
            left = y - pad
            break
    for y in reversed(range(imw)):
        if np.sum(mask[:,y]) > 0:
            right = y + pad
            break
    print(f"top {top}\nbot {bot}\nleft {left}\nright {right}\n")
    # test_img = np.zeros((imh, imw))
    # test_img[top:bot,left:right] = 100
    # print(test_img)
    # print(f"sum {np.sum(test_img)}")
    # plt.imshow(test_img)
    # plt.savefig(f"test bbox.png")
    
    mask_width = np.abs(right - left)
    mask_height = np.abs(bot - top)

    all_v = np.zeros((mask_height, mask_width, cn))

# if abs(s_i - s_j) >= abs (t_i - t_j):
#     d_ij = s_i - s_j
# else:
#     d_ij = t_i - t_j

    for channel in range(3):
        im2var = np.arange(mask_width * mask_height).reshape((mask_height, mask_width)).astype(int) 

        total_len = mask_width * mask_height
        A = np.zeros((total_len * 4, total_len ))
        b = np.zeros(total_len * 4 )
        e = 0
        for y in range(mask_height):
            for x in range(mask_width):
                
                for ii in get_surrounding([y,x]):
                    if mask[top + ii[0], left + ii[1], 0]:
                    # print(f"e {e}")
                        A[e, im2var[ii[0], ii[1]]] = -1
                        A[e, im2var[y, x]] = 1
                        # print(fg[ii[0], ii[1], channel] - fg[y, x, channel])
                        # if abs(fg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel]) >= abs(bg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel])
                        b[e] = max(fg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel],bg[top + y, left + x, channel] - bg[top + ii[0], left + ii[1], channel] )


                    else:
                        A[e, im2var[y, x]] = 1
                        b[e] = bg[top + ii[0], left + ii[1], channel]
                        
                    e +=1
    
        print(f"A.shape {A.shape}\nb.shape {b.shape}\n")
    
        A = csc_matrix(A)

        v = lsqr(A, b, show=False)[0]
        v = v.reshape((mask_height, mask_width))
        print(f"v.shape {v.shape}")
        all_v[:, :, channel] = v
    # bg[np.where(mask == 255)] = all_v[np.where(mask == 255)]
    ans = bg.copy()
    ans[top :bot, left :right] = all_v







    return ans
    return fg * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    # top = None
    # bot = None
    # left = None
    # right = None
    # pad = 5
    # for y in range(imh):
    #     if np.sum(mask[y]) > 0:
    #         top = y - pad
    #         break
    # for y in reversed(range(imh)):
    #     if np.sum(mask[y]) > 0:
    #         bot = y + pad
    #         break
    # for y in range(imw):
    #     if np.sum(mask[:,y]) > 0:
    #         left = y - pad
    #         break
    # for y in reversed(range(imw)):
    #     if np.sum(mask[:,y]) > 0:
    #         right = y + pad
    #         break
    # print(f"top {top}\nbot {bot}\nleft {left}\nright {right}\n")
    # test_img = np.zeros((imh, imw))
    # test_img[top:bot,left:right] = 100
    # print(test_img)
    # print(f"sum {np.sum(test_img)}")
    # plt.imshow(test_img)
    # plt.savefig(f"test bbox.png")
    
    # mask_width = np.abs(right - left)
    # mask_height = np.abs(bot - top)
    imh,imw, cn = rgb_image.shape
    all_v = np.zeros((imh, imw, 1))

# if abs(s_i - s_j) >= abs (t_i - t_j):
#     d_ij = s_i - s_j
# else:
#     d_ij = t_i - t_j

    # for channel in range(3):
    im2var = np.arange(imh * imw).reshape((imh, imw)).astype(int) 

    total_len = imh * imw
    A = np.zeros((total_len * 5, total_len ))
    b = np.zeros(total_len * 5 )
    e = 0
    for y in range(imh):
        for x in range(imw):
            
            for ii in get_surrounding([y,x]):
                # if mask[ii[0], ii[1], 0]:
                # print(f"e {e}")
                if ii[0]>=0 and ii[0]<imh and ii[1]>=0 and ii[1]<imw:
                    A[e, im2var[ii[0], ii[1]]] = -1
                    A[e, im2var[y, x]] = 1
                    # print(fg[ii[0], ii[1], channel] - fg[y, x, channel])
                    # if abs(fg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel]) >= abs(bg[top + y, left + x, channel] - fg[top + ii[0], left + ii[1], channel])
                    # b[e] = max(rgb_image[y, x, 0] - rgb_image[ii[0],ii[1], 0],rgb_image[y, x, 1] - rgb_image[ii[0],ii[1], 1], rgb_image[y, x, 2] - rgb_image[ii[0],ii[1], 2] )
                    # print(b[e])
                    b[e] = rgb_image[y, x, 0] - rgb_image[ii[0],ii[1], 0]

                    # else:
                    #     A[e, im2var[y, x]] = 1
                    #     b[e] = bg[top + ii[0], left + ii[1], channel]
                        
                    e +=1
            A[e, im2var[y, x]] = 1
            b[e] = np.mean(rgb_image[y, x,0])    
            e +=1          
    # A[e, im2var[int(y/2), int(x/2)]] = 1
    # b[e] = np.mean(rgb_image[int(y/2), int(x/2),:])
    
    print(f"A.shape {A.shape}\nb.shape {b.shape}\n")

    A = csc_matrix(A)

    v = lsqr(A, b, show=False)[0]
    # print(v)
    # mean = np.mean(v)
    # std = np.std(v)
    # diff_arr = max(v) - min(v)
    # v = (v - min(v))/diff_arr * 255
    v = v.reshape((imh, imw))
    print(v)

    # print(f"v.shape {v.shape}")
    # all_v[:, :, channel] = v
    # # bg[np.where(mask == 255)] = all_v[np.where(mask == 255)]
    # ans = bg.copy()
    # ans[top :bot, left :right] = all_v







    return v
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    s = img[:, :, 1].reshape((img.shape[0], img.shape[1], 1)) / 255.
    v = img[:, :, 2].reshape((img.shape[0], img.shape[1], 1)) / 255.
    mask = np.ones_like(img[:, :, 0]).reshape((img.shape[0], img.shape[1], 1))
    # mask[0, :] = 0
    # mask[-1, :] = 0
    # mask[:, 0] = 0
    # mask[:, -1] = 0
    # mask = mask > 0
    output = mixed_blend(s, mask, v)
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
        ratio = 1.0
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
        plt.savefig(f"Poisson Blend.png")

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
        plt.savefig(f"Mixed Blend.png")

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        print(f"rgb_image.shape {rgb_image.shape}")
        cv_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(141)
        plt.imshow(rgb_image)
        plt.title('rgb_image')
        plt.subplot(142)
        plt.imshow(cv_gray, cmap='gray')
        plt.title('cv_gray')
        plt.subplot(143)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(144)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
