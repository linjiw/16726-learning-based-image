# Nikhil Uday Shinde
# https://github.com/nikhilushinde/cs194-26_proj3_2.2

# masking_code.py
For masking_code.py call save_masks with image names that you want to create a mask for:

PART 1

in first pop up window click ‘p’ to enter polygon mode this will allow you to select a polygon by clicking various points

when done selecting the polygon click ‘q’ which will fill in the mask on the image for you to see

then click some point in the center that will be used to align the mask with the second image

hit escape when done

the state of the image can be reset by hitting ‘r’ note that you will need to hit ‘p’ again to enter polygon mode

hit escape when done with the first image

additional controls:

click ‘o’ or ‘i’ to rotate the image
click ‘=‘ or ‘-‘ to resize the image NOTE you must use the saved new source image for the mask to be applicable
PART 2:

click anywhere in the image to overlay mask

hit escape when done to save masks

click r at anytime to reset frame

PART 3:

masks are stored with the same name + “_mask.png” in the same folder as the code
new source image is stored with name + “_newsource.png”
NOTE: May need to manually resize images so cv2.imshow can show whole image

# proj2_starter.py
For proj2_starter.py, it storeed all tool functions and the accept arguments as input.

## toy_recon
toy_recon reconstruct a image using poisson blending and maintain the gradient and the top pixel intensity.

## poisson_blend
poisson_blend construct a searching window given the masks, and build a sparse matrix by building each pixel as a equation.

## mixed_blend

mixed_blend maintain the largest gradient in the first half of equation.

## color2gray
call the cv2.cvtColor function.

## mixed_grad_color2gray
call poisson blending for the most clear channels in the HSV space.
