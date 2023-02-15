import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform
from skimage import img_as_ubyte
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import time
import datetime


def get_contrast(b):
    max_b = np.max(b)
    min_b = np.min(b)

    print(max_b)
    print(min_b)

def get_crop_ratio(img):
    img = img_as_ubyte(img)
    edges = cv.Canny(img,100,200)
    height = edges.shape[0]
    width = edges.shape[1]
    def get_ratio(height, width, dim = 0):

        horizontal_val_lst = np.sum(edges,axis= dim)/255/width
        max_args = np.argsort(horizontal_val_lst)
        ratio = min((max_args[-1], height - max_args[-1])) / height

        return ratio
    max_ratio = max(get_ratio(height, width, dim = 1) , get_ratio(width,height,  dim = 0))

    return min(max_ratio,0.10)


def mv_img(img, xx, yy):
    img1 = np.roll(img,xx, axis = 0)
    img2 = np.roll(img1,yy, axis = 1)
    return img2

def cmaes(img1, img2, dim, canny = True, init_xx = 0, init_yy = 0, sigma = 20, population_size = 50, num_iter=1, method = 'SSD'):
  """Optimizes a given function using CMA-ES.

  Args:
    fn: A function that takes as input a vector and outputs a scalar value.
    dim: (int) The dimension of the vector that fn expects as input.
    num_iter: (int) Number of iterations to run CMA-ES.

  Returns:
    mu_vec: An array of size [num_iter, dim] storing the value of mu at each
      iteration.
    best_sample_vec: A list of length [num_iter] storing the function value
      for the best sample from each iteration of CMA-ES.
    mean_sample_vec: A list of length [num_iter] storing the average function
      value across samples from each iteration of CMA-ES.
  """
  # Canny Edge
  if canny:
    img1 = img_as_ubyte(img1)
    img1 = cv.Canny(img1,100,200)
    img2 = img_as_ubyte(img2)
    img2 = cv.Canny(img2,100,200)
  # Hyperparameters
  sigma = sigma
  population_size = population_size
  p_keep = 0.20  # Fraction of population to keep
  noise = 0.25  # Noise added to covariance to prevent it from going to 0.


  # Initialize the mean and covariance
  mu = np.zeros(dim)
  mu[0] += init_xx
  mu[1] += init_yy
  cov = sigma**2 * np.eye(dim)

  mu_vec = []
  best_sample_vec = []
  mean_sample_vec = []
  folder_name = f"./results/{sigma}"
  try:
      os.mkdir(folder_name)
  except:
    pass
  for t in range(num_iter):

    points = np.random.multivariate_normal(mu, cov, population_size)


    points = points.astype(int)

    # xx yy 
    if method == 'SSD':

        values = [SSD(mv_img(img1,  point[0], point[1]), img2) for point in points]
    
    if method == 'NCC':

        values = [NCC(mv_img(img1, point[0], point[1]), img2) for point in points]


    args = np.argsort(values)

    sorted_mat = points[args]

    number_kept = int(population_size * p_keep)



    kept_mat = sorted_mat[-number_kept:,:]

    best_sample_vec.append(values[args[-1]])
    mean_sample_vec.append(np.mean(values))
    

    cov = np.cov(kept_mat.T) + noise * np.eye(dim)



    mu = np.mean(kept_mat,axis=0)

    
    mu = mu.astype(int)


    mu_vec.append(mu)
  print(f"mu_vec {mu_vec}" )
  print(f"best_sample_vec {best_sample_vec}" )
  print(f"mean_sample_vec {mean_sample_vec}" )




  return mu_vec, best_sample_vec, mean_sample_vec

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

# Sum of Squared Differences (SSD) : 
#   sum(sum((image1-image2).^2)) 
# normalized cross-correlation (NCC): 
#   a dot product between two normalized vectors: (image1./||image1|| and image2./||image2||)



def SSD(img1, img2):
    diff = img1 - img2
    return -np.linalg.norm(diff)
    pass

def NCC(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    norm1 = np.linalg.norm(img1)
    norm2 = np.linalg.norm(img2)
    normalized_1 = (img1 - mean1) / norm1
    normalized_2 = (img2 - mean2) / norm2
    res = normalized_1 * normalized_2

    return np.mean(res)

    pass


def align(channel_1, channel_2, extra_channel, square_limit,canny = True, init_xx = 0, init_yy = 0, sigma = 20, population_size = 50,num_iter=50, CMEAS = True, method = 'SSD', min_xx = -40, min_yy = -40, max_xx = 40, max_yy = 40, show_plot = False):



    square_limit = square_limit
    if method == 'SSD':
        score = -np.inf
    
    elif method == 'NCC':
        score = -np.inf
    score_lst = []
    res = [0 , 0]
    gif_movement = [[0,0]]

    xx_range = np.arange(-square_limit,square_limit)
    yy_range = np.arange(-square_limit,square_limit)

    folder_name_0 = f"./results/{sigma}_0"
    try:
      os.mkdir(folder_name_0)
    except:
      pass
    folder_name_1 = f"./results/{sigma}_1"
    try:
      os.mkdir(folder_name_1)
    except:
      pass
    
    if CMEAS:
        mu_vec, best_sample_vec, mean_sample_vec = cmaes( channel_1, channel_2, canny = canny, init_xx = init_xx, init_yy = init_yy, dim = 2,sigma = sigma, population_size = population_size, num_iter=num_iter, method = method)
   
        xx = mu_vec[-1][0]
        yy = mu_vec[-1][1]


        result_img = mv_img(channel_1, xx ,yy)
        gif_movement.append([xx, yy])

        return result_img, gif_movement
    print(f"start with : {init_xx} {init_yy}")
    print(f"xx_range : {xx_range}")
    for xx in xx_range:
        img1 = np.roll(channel_1,init_xx + xx, axis = 0)
        for yy in yy_range:
            img1 = np.roll(img1, init_yy + yy, axis = 1)

            if method == 'SSD':
                tmp = SSD(img1, channel_2)
                if tmp > score:
                    score = tmp
                    res = [init_xx+ xx , init_yy + yy]
                    print(f"updated: {score}")
                    score_lst.append(score)
                    gif_movement.append(res)

                    
            elif method == 'NCC':
                tmp = NCC(img1, channel_2)
                if tmp > score:
                    score = tmp
                    res = [init_xx+ xx , init_yy + yy]
                    print(f"updated: {res}")
                    print(f"updated: {score}")
                    score_lst.append(score)
                    gif_movement.append(res)
    
    result_img = np.roll(channel_1,res[0], axis = 0)
    result_img = np.roll(result_img,res[1], axis = 1)
    gif_movement[-1] = res
    if show_plot:
        plt.plot(score_lst)
        # plt.show()
    print(f"final: {res}")
    return result_img , gif_movement


def crop_border(img, crop_ratio):
    height_crop = int(img.shape[0] * crop_ratio)
    width_crop = int(img.shape[1] * crop_ratio)

    res = img[height_crop:-height_crop, width_crop:-width_crop]


    return res



def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.jpg")]
    frame_one = frames[0]
    frame_one.save(f"{frame_folder}/anime.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def getGIF(img_name,method, gb_mv, rb_mv, r, g ,b):
    count = 0 
    result_img_g = None
    result_img_r = None
    folder_name = f"./results/{img_name}_{method}"
    try:
        os.mkdir(folder_name)
    except:
        pass


    for i in range(len(gb_mv)):
        result_img_g = np.roll(g,gb_mv[i][0], axis = 0)
        result_img_g = np.roll(result_img_g,gb_mv[i][1], axis = 1)
        im_out = np.dstack([r, result_img_g, b])
        fname = f'./results/{img_name}_{method}/{count+i+1}.jpg' 
        skio.imsave(fname, im_out)

    for i in range(len(rb_mv)):
        result_img_r = np.roll(r, rb_mv[i][0],  axis = 0)
        result_img_r = np.roll(result_img_r, rb_mv[i][1], axis = 1)
        im_out = np.dstack([result_img_r, result_img_g, b])
        fname = f'./results/{img_name}_{method}/{count+len(gb_mv)+i+1}.jpg' 
        skio.imsave(fname, im_out)
    make_gif(folder_name)
    


    return True
def RGB_synthesis(folder_name, img_name, square_limit,CROP = True, canny = True, pyramid = True,CMEAS =True , method = 'SSD', min_xx = -40, min_yy = -40, max_xx = 40, max_yy = 40):
    # name of the input file
    im_pth = f"./data/{img_name}" 

    # read in the image
    im = skio.imread(im_pth)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
    
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    if CROP:


        all_ratio = max([get_crop_ratio(b), get_crop_ratio(g),get_crop_ratio(r)])


        b = crop_border (b, all_ratio)
        g = crop_border (g, all_ratio)
        r =  crop_border (r, all_ratio)


    
    start_time = time.time()


    ag = None
    gb_mv = None
    ar = None
    rb_mv = None
    pyramid_b = b.copy()
    pyramid_g = g.copy()
    pyramid_r = r.copy()
    init_xx_g = 0
    init_yy_g = 0
    init_xx_r = 0
    init_yy_r = 0
    scale_lst = [10, 5, 1]
    reverse_scale_lst = reversed(scale_lst)
    times_lst = [2, 5, 1]

    num_iter = [10,5,2]
    sigma = None
    if pyramid:
        for ii, i in enumerate(scale_lst):

            trans_scale = 1/i
            b_ = sk.transform.rescale(pyramid_b, trans_scale)
            g_ = sk.transform.rescale(pyramid_g, trans_scale)
            r_ = sk.transform.rescale(pyramid_r, trans_scale)

            sigma = square_limit / 2
            population_size = int(square_limit) * 4

            if not CMEAS:

                square_limit_per = int(square_limit / scale_lst[0])
            else:
                square_limit_per = square_limit

            all_gif_lst_r = []
            all_gif_lst_g = []
            ag, gb_mv = align(g_, b_, r_ , canny = canny,square_limit = square_limit_per, init_xx=init_xx_g, init_yy=init_yy_g, sigma = sigma, population_size = population_size,num_iter = num_iter[ii], CMEAS = CMEAS, method = method)
            ar, rb_mv = align(r_, b_, g_, canny = canny,square_limit = square_limit_per, init_xx=init_xx_r, init_yy=init_yy_r, sigma = sigma, population_size = population_size,num_iter = num_iter[ii], CMEAS = CMEAS, method = method)



            
            print(f"gb_mv {gb_mv}")
            init_xx_g = int(gb_mv[-1][0] * times_lst[ii])
            init_yy_g = int(gb_mv[-1][1] * times_lst[ii])
            init_xx_r = int(rb_mv[-1][0] * times_lst[ii])
            init_yy_r = int(rb_mv[-1][1] * times_lst[ii])

            print(f"x: g to b {gb_mv[-1][0]}")
            print(f"y: g to b {gb_mv[-1][1]}")
            print(f"x: r to b {rb_mv[-1][0]}")
            print(f"y: r to b {rb_mv[-1][1]}")
            im_out = np.dstack([ar, ag, b_])
            # plt.imshow(im_out)
            # plt.title(f'pyramid, scale = {trans_scale}')
            # plt.show()




            if CMEAS:
                square_limit = int((square_limit * trans_scale + 1) *2 )
            
    else:


        population_size = int(square_limit)
        num_iter = 5

        ag, gb_mv = align(g, b ,r, canny = canny, square_limit = square_limit,sigma = sigma, population_size = population_size,num_iter = num_iter, CMEAS = CMEAS, method = method, min_xx = min_xx, min_yy = min_yy, max_xx = max_xx, max_yy = max_yy)
        ar, rb_mv = align(r, b ,g, canny = canny, square_limit = square_limit,sigma = sigma, population_size = population_size,num_iter = num_iter, CMEAS = CMEAS, method = method, min_xx = min_xx, min_yy = min_yy, max_xx = max_xx, max_yy = max_yy)

    
    end_time = time.time()
    time_str = f"time_cost:{end_time - start_time}"
    print(time_str)

    im_out = np.dstack([ar, ag, b])
    if CROP:
        ratio = get_crop_ratio(im_out)
    # print(f"ratio {ratio}")
        im_out = crop_border(im_out, ratio)
    # plt.imshow(new_img)
    # plt.show()
    # print(f"")
    fname = f'{folder_name}/{img_name}_square_limit{square_limit}_CROP{CROP}_Canny{canny}_Pyramid{pyramid}_CMEAS{CMEAS}_method{method}_shift[{gb_mv[-1]},{rb_mv[-1]}]_{time_str}.jpg' 
    skio.imsave(fname, im_out)

    # skio.imshow(im_out)
    # skio.show()

    