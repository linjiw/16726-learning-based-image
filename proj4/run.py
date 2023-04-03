from re import L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss
from datetime import datetime
import torchvision.transforms as transforms
import math
import wandb

"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
# content_layers_default = ['conv_3']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_weight = 10000.0
content_weight = 1.0
num_steps = 300
unloader = transforms.ToPILImage()  # reconvert into PIL image

def wandb_save(tensor, logname, iter_num):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    images = wandb.Image(image, caption=f'{logname}-{iter_num}.png')       
    wandb.log({f'{logname}': images})
    # plt.imshow(image)
    # plt.savefig(f"{pth}")
    # plt.close()

def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial
    # model = nn.Sequential([])
    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)
    conv_num = 0
    relu_num = 0
    maxpool_num = 0
    content_num = 0
    style_num = 0
    # print(f"cnn {cnn}")
    # print(f"len(cnn.children()) {len(cnn.children())}")
    for i, layer in enumerate(cnn.children()):
        print(f"{i}:{layer} {type(layer)}")
        if isinstance(layer, nn.Conv2d):
            conv_num += 1
            name = f"conv_{conv_num}"
        elif isinstance(layer, nn.ReLU):
            relu_num += 1
            name = f"relu_{relu_num}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            maxpool_num += 1
            name = f"maxpool_{maxpool_num}"
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            # init ContentLoss with target
            content_loss = ContentLoss(target)
            content_num +=1
            model.add_module(f'content_loss_{content_num}', content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            # init ContentLoss with target_feature
            style_loss = StyleLoss(target_feature)
            style_num += 1
            model.add_module(f'style_loss_{style_num}', style_loss)
            style_losses.append(style_loss)
    
    # print(f"model {model}")

    i = 0
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]



    # normalization = TODO
    # model = nn.Sequential(normalization)

    # raise NotImplementedError()

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, task_name, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000.0, content_weight=1.0, content_layers = content_layers_default, style_layers = style_layers_default):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img, content_layers=content_layers, style_layers=style_layers)
    
    print(f"model: {model}")
    print(f"style_losses: {style_losses}")
    print(f"content_losses: {content_losses}")

    # get the optimizer
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    opt = get_image_optimizer(input_img)


    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    # def closure():
    # here
    # which does the following:
    # clear the gradients
    # compute the loss and it's gradient
    # return the loss

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step
    run = [0]
    wandb_save(input_img, 'Style Transfer', run[0])

    while run[0] <= num_steps:

        def closure():
            # clear the gradients
            input_img.grad = None
            
            # compute the loss and it's gradient
            # # model_output = model(input_img)
            # style_score = 0
            # content_score = 0
            # tmp_model = nn.Sequential()
            # x = copy.deepcopy(input_img)
            # for i, layer in enumerate(model.children()):
            #     x = layer(x)

            #     if use_style and isinstance(layer, StyleLoss):
            #         style_score += layer.loss
            #     elif use_content and isinstance(layer, ContentLoss):
            #         content_score += layer.loss
            with torch.no_grad():
                input_img.clamp_(0, 1)

            opt.zero_grad()
            model(input_img)
            style_score = 0.0
            content_score = 0.0
            # since loss is stored at every Loss layer as an attribute.
            # we can access them very easily.
            if use_style:
                for sl in style_losses:
                    style_score += sl.loss
                style_score *= style_weight

            if use_content:
                for cl in content_losses:
                    content_score += cl.loss
                content_score *= content_weight

            loss = style_score + content_score
            loss.backward()


            # style_score *= style_weight
            # content_score *= content_weight
            
            # loss = style_score + content_score
            # loss.backward()

            run[0] += 1
            if run[0] % 20 == 0 and run[0] // 20 >= 1:
                print(f"run {run}:")
                print(f"Style Loss : {style_score:.4f} Content Loss: {content_score:.4f}")
                wandb.log({'Style Loss': style_score,'Content Loss': content_score} )

                print()
                wandb_save(input_img, task_name, run[0])
            else:
                wandb_save(input_img, task_name, run[0])


            return loss
        opt.step(closure)
    
    with torch.no_grad():
        input_img.clamp_(0, 1)

    # make sure to clamp once you are done

    return input_img


def main(style_img_path, content_img_path, task_num, toGIF):
    import os
    style_img_name = os.path.splitext(os.path.basename(style_img_path))[0]
    content_img_name = os.path.splitext(os.path.basename(content_img_path))[0]
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    name_lst = ['reconstruct', 'synthesize', 'transfer']
    task_num = int(task_num)
    toGIF = int(toGIF)
    task_name = name_lst[task_num]
    # my_list = ['apple', 'banana', 'orange']
    content_layers_default = ['conv_4']
    # content_layers_default = ['conv_3']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_string = '_'.join(content_layers_default)
    style_string = '_'.join(style_layers_default)

    # print(my_string)


    wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')

    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # style_img.size == content_img.size
    if style_img.size(dim=1) != 3:
        style_img = style_img.repeat(1, 3, 1, 1)
    if content_img.size(dim=3) > style_img.size(dim=3):
        pad_size = math.ceil((content_img.size(dim=3) - style_img.size(dim=3)) / 2.0)
        padding = transforms.Pad((pad_size, pad_size), padding_mode="reflect")
        style_img = padding(style_img)
    if content_img.size(dim=2) > style_img.size(dim=2):
        pad_size = math.ceil((content_img.size(dim=2) - style_img.size(dim=2)) / 2.0)
        padding = transforms.Pad((0, 0, pad_size, pad_size), padding_mode="reflect")
        style_img = padding(style_img)
        # style_img.resize((1, 3, -1, content_img.size(dim=3)))

    centerCrop = transforms.CenterCrop((content_img.size(dim=2), content_img.size(dim=3)))
    style_img = centerCrop(style_img)
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # interative MPL
    plt.ion()

    # plot the original input image:
    plt.figure()
    imshow(style_img, title=f'/{task_name}/Style Image.png')
    # plt.savefig("style_img.png")
    wandb_save(style_img, 'Style Image', 0)
    plt.close()
    plt.figure()
    imshow(content_img, title=f'/{task_name}/Content Image.png')
    # plt.savefig("content_img.png")
    wandb_save(content_img, 'Content Image', 0)
    plt.close()


    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()



    # image reconstruction
    """
    IMAGE RECONSTRUCTION
    """
    if task_name == 'reconstruct':
        print("Performing Image Reconstruction from white noise initialization")

        
        """
        Report the effect of optimizing content loss at different layers. [15 points]

        """
        
        # input_img = random noise of the size of content_img on the correct device
        # output = reconstruct the image from the noise
        print('Enter your layer, 0 for show results from all layer')
        x = input()
        print('Show results of ' + x)
        target_layer = int(x)
        if x == 0:

            for i in range(18):

                content_layers_default = [f'conv_{i+1}']
                content_string = '_'.join(content_layers_default)
                style_string = '_'.join(style_layers_default)
                wandb.finish()
                wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')


                input_img = torch.randn(content_img.size()).to(device)

                # output = reconstruct the image from the noise
                output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=False, num_steps=300, content_layers=content_layers_default)
                plt.figure()
                now = datetime.now()
                formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
                f_name = f"/{task_name}/{formatted_now}.png"
                imshow(output, title=f_name)
                plt.close()
        else:
            i = x
            content_layers_default = [f'conv_{i}']
            content_string = '_'.join(content_layers_default)
            style_string = '_'.join(style_layers_default)
            wandb.finish()
            wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')

            wandb_save(content_img, 'Content Image', 0)
            wandb_save(style_img, 'Style Image', 0)
            input_img = torch.randn(content_img.size()).to(device)

            # output = reconstruct the image from the noise
            output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=False, num_steps=300, content_layers=content_layers_default)
            plt.figure()
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
            f_name = f"/{task_name}/{formatted_now}.png"
            imshow(output, title=f_name)
            plt.close()

    # plt.savefig("content_img.png")
    # # texture synthesis
    if task_name == 'synthesize':

        # print("Performing Texture Synthesis from white noise initialization")
        # input_img = random noise of the size of content_img on the correct device
        


        
        
        print('Enter your layer, ALL for show results from all layer')
        x = input()
        print('Show results of ' + x)
        # target_layer = int(x)
        
        if x == 'ALL':
            print(f"Show ALL experiments")
            for i in range(11):

                style_layers_default = [f'conv_{i+1}', f'conv_{i+2}', f'conv_{i+3}', f'conv_{i+4}', f'conv_{i+5}']
                content_string = '_'.join(content_layers_default)
                style_string = '_'.join(style_layers_default)
                wandb.finish()
                wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')
                wandb_save(content_img, 'Content Image', 0)
                wandb_save(style_img, 'Style Image', 0)

                input_img = torch.randn(content_img.size()).to(device)

                # output = reconstruct the image from the noise
                output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=True, num_steps=300, content_layers=content_layers_default, style_layers=style_layers_default)
                plt.figure()
                now = datetime.now()
                formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
                f_name = f"/{task_name}/{formatted_now}.png"
                imshow(output, title=f_name)
                plt.close()
        else:
            i = x
            for ii in range(5):
                style_layers_default.append(f"conv_{ii+1 + int(i)}")
            # style_layers_default = [f'conv_{i+1}', f'conv_{i+2}', f'conv_{i+3}', f'conv_{i+4}', f'conv_{i+5}']
            content_string = '_'.join(content_layers_default)
            style_string = '_'.join(style_layers_default)
            wandb.finish()
            wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')
            wandb_save(content_img, 'Content Image', 0)
            wandb_save(style_img, 'Style Image', 0)

            input_img = torch.randn(content_img.size()).to(device)

            # output = reconstruct the image from the noise
            # output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=True, num_steps=300, content_layers=content_layers_default)
            output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=True, num_steps=300, content_layers=content_layers_default, style_layers=style_layers_default)

            plt.figure()
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
            f_name = f"/{task_name}/{formatted_now}.png"
            imshow(output, title=f_name)
            plt.close()
   
    """
    STYLE TRANSFER
    """
    if name_lst[task_num] == 'transfer':

        # center crop style_image to match content image
 


        print('Enter your layer, ALL for show results from all layer')
        x = input()
        print('Show results of ' + x)
        # target_layer = int(x)
        input_img = content_img.clone()

        if x == 'ALL':
            print(f"Show ALL experiments")
            for j in range(16):
                content_layers_default = [f'conv_{j+1}']
                for i in range(11):

                    style_layers_default = [f'conv_{i+1}', f'conv_{i+2}', f'conv_{i+3}', f'conv_{i+4}', f'conv_{i+5}']
                    content_string = '_'.join(content_layers_default)
                    style_string = '_'.join(style_layers_default)
                    wandb.finish()
                    wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')
                    wandb_save(content_img, 'Content Image', 0)
                    wandb_save(style_img, 'Style Image', 0)

                    # input_img = torch.randn(content_img.size()).to(device)
                    input_img = content_img.clone()
                    # output = reconstruct the image from the noise
                    output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=True, num_steps=300, content_layers=content_layers_default, style_layers=style_layers_default)
                    plt.figure()
                    now = datetime.now()
                    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
                    f_name = f"/{task_name}/{formatted_now}.png"
                    imshow(output, title=f_name)
                    plt.close()
        else:
            # i = x
            i = input("style start layer")
            j = input("content start layer")
            i = int(i)
            j = int(j)

            content_layers_default = [f'conv_{j+1}']
            # style_layers_default = [f'conv_{i+1}', f'conv_{i+2}', f'conv_{i+3}', f'conv_{i+4}', f'conv_{i+5}']
            style_layers_default = []
            for ii in range(5):
                style_layers_default.append(f"conv_{ii + 1 +i}")

            content_string = '_'.join(content_layers_default)
            style_string = '_'.join(style_layers_default)
            wandb.finish()
            wandb.init(project="hw4-StyleOptimization", name=f'{task_name}_{content_string}_{style_string}_{style_img_name}_{content_img_name}_{formatted_now}')
            wandb_save(content_img, 'Content Image', 0)
            wandb_save(style_img, 'Style Image', 0)

            # input_img = torch.randn(content_img.size()).to(device)
            input_img = content_img.clone()
            # output = reconstruct the image from the noise
            # output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=True, num_steps=300, content_layers=content_layers_default)
            output = run_optimization(cnn, task_name , content_img, style_img, input_img, use_style=True, num_steps=300, content_layers=content_layers_default, style_layers=style_layers_default)

            plt.figure()
            now = datetime.now()
            formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
            f_name = f"/{task_name}/{formatted_now}.png"
            imshow(output, title=f_name)
            plt.close()

    plt.ioff()


    plt.show()


    import os
    from PIL import Image

    if toGIF:
        # Path to the folder containing the image files
        folder_path = '/content/drive/MyDrive/16726/Computer-vision-learning/hw4/wandb/latest-run/files/media/images'

        # List all image files in the folder
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.png')]

        # Sort the file names alphabetically
        file_names.sort()
        print(f"len(file_names) {len(file_names)}")
        # Create an array of images from the files
        images = [Image.open(os.path.join(folder_path, f)) for f in file_names]

        # Save the animation as a GIF file
        images[0].save(f'{style_img_name}_{content_img_name}_{formatted_now}.gif', save_all=True, append_images=images[1:], duration=400, loop=0)
    else:
        pass

if __name__ == '__main__':
    args = sys.argv[1:5]
    main(*args)
