# README

## Style Optimization - Course Project

### Overview
This course project focuses on style optimization using neural style transfer. The project includes three main tasks: image reconstruction, texture synthesis, and style transfer. It allows the user to experiment with different layers in a VGG19 model to analyze the effects of content and style optimization.

### Main Function
The `main` function is the central function in this project. It takes four arguments: `style_img_path`, `content_img_path`, `task_num`, and `toGIF`.

#### Arguments
- `style_img_path` (str): Path to the style image file.
- `content_img_path` (str): Path to the content image file.
- `task_num` (int): Task number (0 for image reconstruction, 1 for texture synthesis, 2 for style transfer).
- `toGIF` (int): If set to 1, the function will generate a GIF of the optimization process.

#### Usage
The function starts by loading the style and content images, initializing the pre-trained VGG19 model, and setting up the Weights & Biases (wandb) project for logging.

Based on the `task_num` argument, the function performs one of the following tasks:

1. **Image Reconstruction**: Reconstructs the content image from random noise using content loss optimization. The user can specify different layers for optimization or run experiments on all layers.

2. **Texture Synthesis**: Synthesizes a texture from the style image using style loss optimization. The user can specify different layers for optimization or run experiments on all layers.

3. **Style Transfer**: Transfers the style from the style image to the content image using a combination of content and style loss optimization. The user can specify different layers for optimization.

The function then saves and logs the results using wandb.

#### Example
To use the `main` function, provide the paths to the style and content images, specify the task number, and decide whether to create a GIF. For example:

```python
main("path/to/style_image.jpg", "path/to/content_image.jpg", 2, 0)
