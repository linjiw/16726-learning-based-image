# Project README: When Cats meet GANs
In this project, Linji Wang explores and implements two types of Generative Adversarial Networks (GANs), namely Deep Convolutional GAN (DCGAN) and CycleGAN, to generate grumpy cats and perform image-to-image translation between two types of cats (Grumpy and Russian Blue) and between apples and oranges.

## Part 1: Deep Convolutional GAN
The DCGAN is implemented with advanced data augmentation techniques to generate realistic images of grumpy cats. The data_loader.py file includes a deluxe version of data augmentation, which is used to resize, crop, flip, and normalize the images. The DCDiscriminator and DCGenerator classes in the models.py file are implemented, and the training loop is implemented in the vanilla_gan.py file.

Differentiable Augmentation is applied to the images to achieve faster convergence and stable loss curves. The project shows a comparison between different data preprocessing techniques, i.e., basic, deluxe, and diff_aug, in terms of Discriminator Loss, Generator Loss, Convergence Rate, and Stability. The results suggest that differential augmentations result in the fastest convergence and the most stable loss curves.

## Part 2: CycleGAN
The CycleGAN is implemented to perform image-to-image translation between two types of cats (Grumpy and Russian Blue) and between apples and oranges. The cycle_gan.py file includes the CycleGenerator and CycleDiscriminator classes, and the training loop is implemented. The project demonstrates the use of advanced data augmentation techniques to improve the quality and diversity of generated images.

Overall, this project provides a comprehensive study of GANs and advanced augmentation techniques, making it a useful resource for anyone interested in exploring GANs and image generation.