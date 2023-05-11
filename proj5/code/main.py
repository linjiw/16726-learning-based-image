# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe), modified by Zhiqiu Lin (zl279@cornell.edu)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np
import wandb
from LBFGS import FullBatchLBFGS
from datetime import datetime
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import torchvision.utils as vutils
from torchvision.models import vgg19

from dataloader import get_data_loader
unloader = transforms.ToPILImage()  # reconvert into PIL image

def wandb_save(tensor, logname, iter_num):
    # image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image.squeeze(0)      # remove the fake batch dimension
    # image = unloader(image)
    images = wandb.Image(tensor, caption=f'{iter_num}.png')       
    wandb.log({f'{logname}': images})

def build_model(name):
    if name.startswith('vanilla'):
        z_dim = 100
        model_path = 'pretrained/%s.ckpt' % name
        pretrain = torch.load(model_path)
        from vanilla.models import DCGenerator
        model = DCGenerator(z_dim, 32, 'instance')
        model.load_state_dict(pretrain)

    elif name == 'stylegan':
        model_path = 'pretrained/%s.ckpt' % name
        import sys
        sys.path.insert(0, 'stylegan')
        from stylegan import dnnlib, legacy
        with dnnlib.util.open_url(model_path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
            z_dim = model.z_dim
    else:
         return NotImplementedError('model [%s] is not implemented', name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, z_dim


class Wrapper(nn.Module):
    """The wrapper helps to abstract stylegan / vanilla GAN, z / w latent"""
    def __init__(self, args, model, z_dim, resolution):
        super().__init__()
        self.model, self.z_dim = model, z_dim
        self.latent = args.latent
        self.is_style = args.model == 'stylegan'
        self.resolution = resolution
        self.resize = transforms.Resize((self.resolution, self.resolution))
        print(f"Wrapper.init: latent={self.latent}, z_dim={z_dim}, resolution={resolution}")
    def forward(self, param):
        if self.latent == 'z':
            if self.is_style:
                image = self.model(param, None)
            else:
                image = self.model(param)
        # w / wp
        else:
            assert self.is_style
            # if self.latent == 'w':
            #     print(f"self.model.mapping.num_ws {self.model.mapping.num_ws}")
            #     print(f"before repeating: param.shape {param.shape}")
            #     param = param.repeat(1, self.model.mapping.num_ws, 1)
            #     print(f"after repeating: param.shape {param.shape}")

            image = self.model.synthesis(param)
        image = self.resize(image)
        return image


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # need to detach and cache the appropriate thing
        # self.target = target_feature.detach()
        self.target = gram_matrix(target_feature).detach()

        # raise NotImplementedError()

    def forward(self, input):
        # need to cache the appropriate loss value in self.loss
        # self.loss = TODO
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)        # raise NotImplementedError()
        return input
def gram_matrix(activations):
    a, b, c, d = activations.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = activations.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

    return normalized_gram
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # you need to `detach' the target content from the graph used to
        # compute the gradient in the forward pass that made it so that we don't track
        # those gradients anymore
        # self.target = TODO
        self.target = target.detach()
        # raise NotImplementedError()

    def forward(self, input):
        # this needs to be a passthrough where you save the appropriate loss value
        # self.loss = TODO
        self.loss = F.mse_loss(input, self.target)

        # raise NotImplementedError()
        return input
class PerceptualLoss(nn.Module):
    def __init__(self, add_layer=['conv_5']):
        super().__init__()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        norm = Normalization(cnn_normalization_mean, cnn_normalization_std)
        cnn = vgg19(pretrained=True).features.to(device).eval()
        
        # TODO (Part 1): implement the Perceptual/Content loss
        #                hint: hw4
        # You may split the model into different parts and store each part in 'self.model'


        self.model = nn.ModuleList()
        self.model.add_module("norm", norm)

        i = 0
        conv_num = 0
        for layer in cnn.children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
                # print(f"relu added")
            if isinstance(layer, nn.Conv2d):
                conv_num += 1
                # layer = nn.ReLU(inplace=False)
                # print(f"relu added")
            self.model.add_module(str(i), layer)
            # print(f"conv_{conv_num}")
            if f"conv_{conv_num}" in add_layer:
                self.model.add_module(f"loss_{i}", nn.Identity())
                # print(f"conv_{conv_num}--------added!!")
                break
            i += 1
            # break
        # i = 0
        # for i in range(len(self.model) - 1, -1, -1):
        #     if isinstance(self.model[i], nn.Identity()):
        #         break
        # self.model = self.model[:(i + 1)]      
        print(self.model)
    def forward(self, pred, target):
        mask = None
        if isinstance(target, tuple):
            target, mask = target
        
        # loss = 0.
        loss = torch.tensor(0.0, device=device)

        for net in self.model:
            pred = net(pred)
            target = net(target)

            # TODO (Part 1): implement the forward call for perceptual loss
            #                free feel to rewrite the entire forward call based on your
            #                implementation in hw4
            # TODO (Part 3): if mask is not None, then you should mask out the gradient
            #                based on 'mask==0'. You may use F.adaptive_avg_pool2d() to 
            #                resize the mask such that it has the same shape as the feature map.

            if isinstance(net, nn.Identity):
                l = F.mse_loss(pred, target, reduction='none')
                # print(f"Lp: {l}")
                if mask is not None:
                    resized_mask = F.adaptive_avg_pool2d(mask, pred.shape[-2:])
                    l = l * (resized_mask == 0).type_as(l)
                # loss += l.mean()
                loss = loss.add_(l.mean())
            
            pass
        # loss = torch.tensor(loss)
        return loss

class Criterion(nn.Module):
    def __init__(self, args, mask=False, layer=['conv_5']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        self.l1_wgt = args.l1_wgt # weight for l1 loss/mask loss
        self.mask = mask
        self.reg_wgt = args.reg_wgt
        self.perc = PerceptualLoss(layer)
        self.l1 = nn.L1Loss() 

    def forward(self, pred, target, delta):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        if self.mask:
            target, mask = target
            # TODO (Part 3): loss with mask
            loss = self.perc_wgt * self.perc(pred, (target, mask))
            wandb.log({'Perceptual Loss': loss.item()} )

            l1 = self.l1(pred, target)
            if mask is not None:
                resized_mask = F.adaptive_avg_pool2d(mask, pred.shape[-2:])
                l1 = l1 * (resized_mask == 0).type_as(l1)
            wandb.log({'L1 Loss': (self.l1_wgt * l1.mean()).item()} )

            loss = loss.add_(self.l1_wgt * l1.mean() )

            pass
        else:
            # TODO (Part 1): loss w/o mask
            # return self.perc(pred, target)
            loss = self.perc_wgt * self.perc(pred, target)
            wandb.log({'Perceptual Loss': loss.item()} )
            wandb.log({'L1 Loss': (self.l1_wgt * self.l1(pred, target)).item()} )

            loss = loss.add_(self.l1_wgt * self.l1(pred, target))  


            pass

        delta_l2_norm = torch.norm(delta, p=2)
        wandb.log({'Reg Loss': (delta_l2_norm * self.reg_wgt).item()} ) 
        loss = loss.add_(delta_l2_norm * self.reg_wgt)

        return loss


def save_images(image, fname,idx, col=8):
    image = image.cpu().detach()
    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    wandb_save(image,fname, idx)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, fname, idx, col) for idx, each  in enumerate(image_list)]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    imageio.mimsave(fname + '.gif', image_list)
    # wandb_save(image_list,fname, idx)
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    wandb.log(
  {"video": wandb.Video(fname + '.gif', fps=4, format="gif",caption=f'{formatted_now}.png')})

def sample_noise(dim, device, latent, model, N=1, from_mean=False):
    """
    To generate a noise vector, just sample from a normal distribution.
    To generate a style latent, you need to map the noise (z) to the style (W) space given the `model`.
    You will be using model.mapping for this function.
    Specifically,
    if from_mean=False,
        sample N noise vector (z) or N style latent(w/w+) depending on latent value.
    if from_mean=True
        if latent == 'z': Return zero vectors since zero is the mean for standard gaussian
        if latent == 'w'/'w+': You should sample N=10000 z to generate w/w+ and then take the mean.
    Some hint on the z-mapping can be found at stylegan/generate_gif.py L70:81.
    Additionally, you can look at stylegan/training/networks.py class Generator L477:500
    :return: Tensor on device in shape of (N, dim) if latent == z
             Tensor on device in shape of (N, 1, dim) if latent == w
             Tensor on device in shape of (N, nw, dim) if latent == w+
    """
    # TODO (Part 1): Finish the function below according to the comment above
    print(f"{latent}: {dim}")
    # nw stands for the number of "w" latents in the StyleGAN architecture. 
    # In StyleGAN, the generator uses a series of Adaptive Instance Normalization (AdaIN)
    # layers to control the style of the output image.
    if latent == 'z':
        vector = torch.randn(N, dim, device=device) if not from_mean else torch.zeros(N, dim, device=device)
    elif latent == 'w':
        if from_mean:
            z_10000 = torch.randn(10000, dim, device=device)
            w = model.mapping(z_10000, None)
            vector = w.mean(dim=0, keepdim=True)
        else:
            z = torch.randn(N, dim, device=device)
            print(f"z.shape {z.shape}")
            vector = model.mapping(z, None)
            print(f"before mapping: vector.shape {vector.shape}")
   

    elif latent == 'w+':
        nw = model.num_ws

        if from_mean:
            z_10000 = torch.randn(10000, dim, device=device)
            w = model.mapping(z_10000, None).view(10000, nw, dim)
            vector = w.mean(dim=0, keepdim=True)
        else:
            z = torch.randn(N, dim, device=device)
            vector = model.mapping(z, None).view(N, nw, dim)
    else:
        raise NotImplementedError('%s is not supported' % latent)
    return vector


def optimize_para(wrapper, param, target, criterion, num_step, save_prefix=None, res=False):
    """
    wrapper: image = wrapper(z / w/ w+): an interface for a generator forward pass.
    param: z / w / w+
    target: (1, C, H, W)
    criterion: loss(pred, target)
    """
    print(f"param.shape {param.shape}")
    # print(f"target.shape {target.shape}")

    delta = torch.zeros_like(param)
    delta = delta.requires_grad_().to(device)
    optimizer = FullBatchLBFGS([delta], lr=.1, line_search='Wolfe')
    iter_count = [0]
    def closure():
        iter_count[0] += 1
        # TODO (Part 1): Your optimiztion code. Free free to try out SGD/Adam.
        optimizer.zero_grad()

        """
        adjusted_param = param + delta: In the optimization process,
         the code tries to find the optimal adjustments (delta) to the 
         original parameter (param) to minimize the loss. This line 
         calculates the adjusted_param by adding the current value of delta
          to the original param. The adjusted_param represents the updated 
          noise vector (z) or style latent vector (w/w+) after applying the adjustments.
        """
        """
        image = wrapper(adjusted_param): The wrapper function is an interface 
        for the generator's forward pass. Given the adjusted parameter, 
        wrapper generates an output image based on the current adjusted noise vector (z)
         or style latent vector (w/w+). The output image is stored in the variable image.
        """
        """
        loss = criterion(image, target): The criterion function is used to calculate 
        the loss between the generated image and the target image. The objective of 
        the optimization process is to minimize this loss by adjusting the input parameters.
         The calculated loss is stored in the variable loss.
        """
        adjusted_param = param + delta
        # print(f"adjusted_param.shape {adjusted_param.shape}")
        image = wrapper(adjusted_param)
        loss = criterion(image, target, delta)
        # if regularization:
        #     reg_weight = 1e-6
        # loss += reg_weight * delta_l2_norm
        # loss.backward()
        wandb.log({'Total Loss': loss.item()} )
        if iter_count[0] % 250 == 0:
            # visualization code
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            

            if save_prefix is not None:
                iter_result = image.data.clamp_(-1, 1)
                save_images(iter_result, save_prefix, iter_count[0])
        return loss

    loss = closure()
    loss.backward()
    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
        
    image = wrapper(param)
    return param + delta, image


def sample(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim, args.resolution)
    batch_size = 16
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    noise = sample_noise(z_dim, device, args.latent, model, batch_size)
    image = wrapper(noise)
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)


def project(args):
    # load images
    # print(f"args {args}")
    # print(f"args.latent {args.latent}")

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    loader = get_data_loader(args.input, args.resolution, is_train=False)
    wandb.init(project="hw5-GAN Photo Editing", name=f'{args.model}_{args.mode}_{args.latent}_Perc{args.perc_wgt}_{args.l1_wgt}_{args.reg_wgt}_{formatted_now}')

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim, args.resolution)
    print('model {} loaded'.format(args.model))
    criterion = Criterion(args)
    # project each image
    for idx, (data, _) in enumerate(loader):
        target = data.to(device)
        save_images(data, 'output/project/%d_data' % idx, idx, 1)
        param = sample_noise(z_dim, device, args.latent, model)
        optimize_para(wrapper, param, target, criterion, args.n_iters,
                      'output/project/%d_%s_%s_%g_%g_%g' % (idx, args.model, args.latent, args.perc_wgt,args.l1_wgt ,args.reg_wgt))
        if idx >= 0:
            break


def draw(args):
    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim, args.resolution)
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="hw5-GAN Photo Editing", name=f'{args.model}_{args.mode}_{args.latent}_Perc{args.perc_wgt}_{args.l1_wgt}_{args.reg_wgt}_{formatted_now}')

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution, alpha=True)
    criterion = Criterion(args, True)
    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb, 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)
        optimize_para(wrapper, param, (rgb, mask), criterion, args.n_iters,
                      'output/project/%d_%s_%s_%g_%g_%g' % (idx, args.model, args.latent, args.perc_wgt,args.l1_wgt ,args.reg_wgt))
        # if idx >= 0
        #     break      

        # TODO (Part 3): optimize sketch 2 image
        #                hint: Set from_mean=True when sampling noise vector


def interpolate(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim, args.resolution)
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
    # loader = get_data_loader(args.input, args.resolution, is_train=False)
    wandb.init(project="hw5-GAN Photo Editing", name=f'{args.model}_{args.mode}_{args.latent}_Perc{args.perc_wgt}_{args.l1_wgt}_{args.reg_wgt}_{formatted_now}')

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution)
    criterion = Criterion(args)
    for idx, (image, _) in enumerate(loader):
        save_images(image, 'output/interpolate/%d' % (idx), idx)
        target = image.to(device)
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)
        param, recon = optimize_para(wrapper, param, target, criterion, args.n_iters,'output/project/%d_%s_%s_%g_%g_%g' % (idx, args.model, args.latent, args.perc_wgt,args.l1_wgt ,args.reg_wgt))
        save_images(recon, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent), idx)
        if idx % 2 == 0:
            src = param
            continue
        # src = param
        dst = param
        alpha_list = np.linspace(0, 1, 50)
        image_list = []
        with torch.no_grad():
            # TODO (Part 2): interpolation code
            #                hint: Write a for loop to append the convex combinations to image_list
            for alpha in alpha_list:
                z_interpolated = alpha * src + (1 - alpha) * dst
                img_interpolated = wrapper(z_interpolated)
                image_list.append(img_interpolated)
            
            pass
        save_gifs(image_list, 'output/project/%d_%s_%s_%g_%g_%g' % (idx, args.model, args.latent, args.perc_wgt,args.l1_wgt ,args.reg_wgt))
        # if idx >= 3:
        #     break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan'])
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='z', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=1000, help="number of optimization steps in the image projection")
    parser.add_argument('--perc_wgt', type=float, default=0.01, help="perc loss weight")
    parser.add_argument('--l1_wgt', type=float, default=10., help="L1 pixel loss weight")
    parser.add_argument('--resolution', type=int, default=64, help='Resolution of images')
    parser.add_argument('--input', type=str, default='data/cat/*.png', help="path to the input image")
    parser.add_argument('--reg_wgt', type=float, default=1e-6, help="regularization loss weight")
 
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    import random
    seed = 1000
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if args.mode == 'sample':
        sample(args)
    elif args.mode == 'project':
        project(args)
    elif args.mode == 'draw':
        draw(args)
    elif args.mode == 'interpolate':
        interpolate(args)
    wandb.finish()
