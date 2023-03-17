from __future__ import print_function

from proj2_starter import *

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
import skimage as sk
import skimage.io as skio
# parser = argparse.ArgumentParser("Poisson blending.")
# parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
# args, _ = parser.parse_known_args()

# # Example script: python proj2_starter.py -q toy

# if args.question == "toy":

image = imageio.imread('./data/toy_problem.png') / 255.
image_hat = toy_recon(image)

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Input')
plt.subplot(122)
plt.imshow(image_hat, cmap='gray')
plt.title('Output')
plt.savefig(f"./data/toy_reconstruction.png")
plt.show()

    # skio.imsave()
    

# Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
# if args.question == "blend":
#     parser.add_argument("-s", "--source", required=True)
#     parser.add_argument("-t", "--target", required=True)
#     parser.add_argument("-m", "--mask", required=True)
#     parser.add_argument("-r", "--ratio", required=True)
#     args = parser.parse_args()

#     # after alignment (masking_code.py)
#     if args.ratio:
        
#         ratio = float(args.ratio)  
#     else:
ratio = 0.5
source = 'data/source_01_newsource.png'
target = 'data/target_01.jpg'
mask = 'data/target_01_mask.png'
fg = cv2.resize(imageio.imread(source), (0, 0), fx=ratio, fy=ratio)
bg = cv2.resize(imageio.imread(target), (0, 0), fx=ratio, fy=ratio)
mask = cv2.resize(imageio.imread(mask), (0, 0), fx=ratio, fy=ratio)

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
name2 = source.split('/')[-1][:-4] + "_" + target.split('/')[-1][:-4] + "Blend.jpg"

plt.savefig(f"./data/{name2}")
plt.show()
    # name2 = name2[:-4]
    

# if args.question == "mixed":
#     parser.add_argument("-s", "--source", required=True)
#     parser.add_argument("-t", "--target", required=True)
#     parser.add_argument("-m", "--mask", required=True)
#     parser.add_argument("-r", "--ratio", required=False)
#     args = parser.parse_args()

#     # after alignment (masking_code.py)
#     if args.ratio:
        
#         ratio = float(args.ratio)
        
#     else:
#         ratio = 1.0
    # fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
    # bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
    # mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

    # fg = fg / 255.
    # bg = bg / 255.
# mask = (mask.sum(axis=2, keepdims=True) > 0)

blend_img = mixed_blend(fg, mask, bg)
plt.subplot(121)
plt.imshow(fg * mask + bg * (1 - mask))
plt.title('Naive Blend')
plt.subplot(122)
plt.imshow(blend_img)
plt.title('Mixed Blend')
name2 = source.split('/')[-1][:-4] + "_" + target.split('/')[-1][:-4] + "Mixed Blend.jpg"

plt.savefig(f"./data/{name2}")
plt.show()
# plt.imsave(f"./data/results_{name2}",blend_img)
skio.imsave(f"./data/results_{name2}",blend_img)

    # name2 = args.source.split('/')[-1][:-4] + "_" + args.target.split('/')[-1][:-4] + "_Mixed Blend.jpg"
    # # name2 = name2[:-4]
    # plt.savefig(f"./data/{name2}")

# if args.question == "color2gray":
#     parser.add_argument("-s", "--source", required=True)
#     args = parser.parse_args()
source = 'data/colorBlindTest35.png'
rgb_image = imageio.imread(source)
print(f"rgb_image.shape {rgb_image.shape}")
cv_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
gray_image = color2gray(rgb_image)
mixed_grad_img = mixed_grad_color2gray(rgb_image)
hsv_image  = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


plt.subplot(231)
plt.imshow(hsv_image[:,:,0])
plt.title('hsv_image 1')
plt.subplot(232)
plt.imshow(hsv_image[:,:,1])
plt.title('hsv_image 2')
plt.subplot(233)
plt.imshow(hsv_image[:,:,2])
plt.title('hsv_image 3')
plt.subplot(234)
plt.imshow(rgb_image, cmap='gray')
plt.title('rgb_image')
plt.subplot(235)
plt.imshow(gray_image, cmap='gray')
plt.title('rgb2gray')
plt.subplot(236)
plt.imshow(mixed_grad_img, cmap='gray')
plt.title('mixed gradient')
plt.savefig(f"./data/rgb2gray_results.jpg")
plt.show()

plt.close()
