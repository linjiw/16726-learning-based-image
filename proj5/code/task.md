# Project Title

Brief introduction to the project and its goals.

## Techniques Used

- Inverting the Generator: A technique to find the latent code that closely reconstructs a given real image using a pre-trained generator.
- Interpolating Cats: Interpolating between two cat images by finding their latent codes and combining them to generate a new image.
- Scribble to Image: Generating realistic images from hand-drawn sketches using optimization and generator networks.

## Part 1: Inverting the Generator

A page showcasing example outputs for different combinations of losses, generative models, and latent spaces. Include commentary on the results and performance.

### Technique Details

- Loss function: Lp losses and perceptual (content) losses.
- Optimization: First-order or quasi-Newton optimization methods (e.g., LBFGS).
- Generative models: Vanilla GAN, StyleGAN.

### Example 1

![Image 1](image_path_1.png)

**Losses:** _Description_

**Generative Model:** _Description_

**Latent Space:** _Description_

**Commentary:** _Description_

### Example 2

![Image 2](image_path_2.png)

**Losses:** _Description_

**Generative Model:** _Description_

**Latent Space:** _Description_

**Commentary:** _Description_

## Part 2: Interpolate your Cats

A page displaying interpolations between grumpy cats and commentary on the image quality and visual interpolation.

### Technique Details

- Inverse latent codes: Finding the inverse latent codes of two images.
- Convex combination: Combining the inverse latent codes using convex combination.
- Generating new images: Generate new images using the combined latent codes.

### Interpolation Example 1

![Interpolation 1](interpolation_path_1.gif)

**Generative Model:** _Description_

**Latent Space:** _Description_

**Commentary:** _Description_

### Interpolation Example 2

![Interpolation 2](interpolation_path_2.gif)

**Generative Model:** _Description_

**Latent Space:** _Description_

**Commentary:** _Description_

## Part 3: Scribble to Image

A page displaying example outputs of cats drawn using the model. Include commentary on the results and the effect of using sparser or denser sketches and colors.

### Technique Details

- Color scribble constraints: Adding constraints based on hand-drawn sketches.
- Penalized nonconvex optimization: Solving a penalized nonconvex optimization problem to generate images subject to constraints.
- Generator network: Using a trained generator network to generate the final images.

### Drawing Example 1

![Drawing 1](drawing_path_1.png)

**Sketch Density:** _Description_

**Colors Used:** _Description_

**Commentary:** _Description_

### Drawing Example 2

![Drawing 2](drawing_path_2.png)

**Sketch Density:** _Description_

**Colors Used:** _Description_

**Commentary:** _Description_

## Conclusion

Summary of the project, challenges faced, and potential future work.
