# CMU 16-726 Learning-Based Image Synthesis / Spring 2023, Assignment 3
#
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the second part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters:
#       python cycle_gan.py
#
#    To train with cycle consistency loss:
#       python cycle_gan.py --use_cycle_consistency_loss
#
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU),
#    then you can obtain better results by
#    increasing the number of filters used in the generator
#    and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import argparse
import os

import imageio
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import numpy as np

import utils
from data_loader import get_data_loader
from models import CycleGenerator, DCDiscriminator, PatchDiscriminator
from diff_augment import DiffAugment
# from diff_augment import DiffAugment
# import wandb
from datetime import datetime
policy = 'color,translation,cutout'

# wandb.login()

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    model_dict = {'cycle': CycleGenerator}
    G_XtoY = model_dict[opts.gen](conv_dim=opts.g_conv_dim, norm=opts.norm)
    G_YtoX = model_dict[opts.gen](conv_dim=opts.g_conv_dim, norm=opts.norm)

    model_dict = {'dc': DCDiscriminator, 'patch': PatchDiscriminator}
    D_X = model_dict[opts.disc](conv_dim=opts.d_conv_dim, norm=opts.norm)
    D_Y = model_dict[opts.disc](conv_dim=opts.d_conv_dim, norm=opts.norm)
    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Save generators G_YtoX, G_XtoY and discriminators D_X, D_Y."""
    G_XtoY_path = os.path.join(
        opts.checkpoint_dir, 'G_XtoY_iter%d.pkl' % iteration
    )
    G_YtoX_path = os.path.join(
        opts.checkpoint_dir, 'G_YtoX_iter%d.pkl' % iteration
    )
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X_iter%d.pkl' % iteration)
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y_iter%d.pkl' % iteration)
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def merge_images(sources, targets, opts, k=10):
    """
    Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)


def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = utils.to_data(fixed_X), utils.to_data(fake_X)
    Y, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(
        opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration)
    )
    merged = np.uint8(255 * (merged + 1) / 2)
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))

    # images = wandb.Image(merged, caption=f'sample-{iteration}-X-Y.png')         
    # wandb.log({'sample-X-Y': images})

    merged = merge_images(Y, fake_X, opts)
    path = os.path.join(
        opts.sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration)
    )
    merged = np.uint8(255 * (merged + 1) / 2)
    imageio.imwrite(path, merged)
    print('Saved {}'.format(path))
    # images = wandb.Image(merged, caption=f'sample-{iteration}-Y-X.png')       
    # wandb.log({'sample-Y-X': images})


def training_loop(dataloader_X, dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """
    now = datetime.now()

    datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")

    # wandb.init(project="hw3-CycleGAN", name=f'{opts.sample_dir}_{datetime_string}')

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
    d_params = list(D_X.parameters()) + list(D_Y.parameters())

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    # Get some fixed data from domains X and Y for sampling.
    # These are images that are held constant throughout training,
    # that allow us to inspect the model's performance.
    fixed_X = utils.to_var(next(iter_X))
    fixed_Y = utils.to_var(next(iter_Y))

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    for iteration in range(1, opts.train_iters + 1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X = next(iter_X)
        images_X = utils.to_var(images_X)

        images_Y = next(iter_Y)
        images_Y = utils.to_var(images_Y)
        # D(DiffAugment(real_images, policy='color,translation,cutout', channels_first=False ))
        # TRAIN THE DISCRIMINATORS
        # 1. Compute the discriminator losses on real images
        if not opts.use_diffaug:
            D_X_loss = torch.mean((D_X(images_X) - 1) ** 2)
            D_Y_loss = torch.mean((D_Y(images_Y) - 1) ** 2)
        else:
            D_X_loss = torch.mean((D_X(DiffAugment(images_X, policy='color,translation,cutout', channels_first=False )) - 1) ** 2)
            D_Y_loss = torch.mean((D_Y(DiffAugment(images_Y, policy='color,translation,cutout', channels_first=False )) - 1) ** 2)


        d_real_loss = D_X_loss + D_Y_loss

        # 2. Generate domain-X-like images based on real images in domain Y
        fake_X = G_YtoX(images_Y).detach()

        # 3. Compute the loss for D_X
        if not opts.use_diffaug:

            D_X_loss = torch.mean((D_X(fake_X) ) ** 2)
        else:
            D_X_loss = torch.mean((D_X(DiffAugment(fake_X, policy='color,translation,cutout', channels_first=False )) ) ** 2)


        # 4. Generate domain-Y-like images based on real images in domain X
        fake_Y = G_XtoY(images_X).detach()

        # 5. Compute the loss for D_Y
        if not opts.use_diffaug:

            D_Y_loss = torch.mean((D_Y(fake_Y) ) ** 2)
        else:
            D_Y_loss = torch.mean((D_Y(DiffAugment(fake_Y, policy='color,translation,cutout', channels_first=False )) ) ** 2)


        d_fake_loss = D_X_loss + D_Y_loss

        # sum up the losses and update D_X and D_Y
        d_optimizer.zero_grad()
        d_total_loss = d_real_loss + d_fake_loss
        d_total_loss.backward()
        d_optimizer.step()

        # plot the losses in tensorboard
        logger.add_scalar('D/XY/real', D_X_loss, iteration)
        logger.add_scalar('D/YX/real', D_Y_loss, iteration)
        logger.add_scalar('D/XY/fake', D_X_loss, iteration)
        logger.add_scalar('D/YX/fake', D_Y_loss, iteration)

        # TRAIN THE GENERATORS
        # 1. Generate domain-X-like images based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        if not opts.use_diffaug:

            g_loss = torch.mean((D_X(fake_X) - 1 ) ** 2)
        else:
            g_loss = torch.mean((D_X(DiffAugment(fake_X, policy='color,translation,cutout', channels_first=False )) - 1 ) ** 2)


        logger.add_scalar('G/XY/fake', g_loss, iteration)

        if opts.use_cycle_consistency_loss:
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.mean((images_X - G_YtoX(G_XtoY(images_X)))**2)


            g_loss += opts.lambda_cycle * cycle_consistency_loss
            logger.add_scalar('G/XY/cycle', opts.lambda_cycle * cycle_consistency_loss, iteration)

        # X--Y-->X CYCLE
        # 1. Generate domain-Y-like images based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain Y
        if not opts.use_diffaug:

            g_loss += torch.mean((D_Y(fake_Y) - 1 ) ** 2)
        else:
            g_loss += torch.mean((D_Y(DiffAugment(fake_Y, policy='color,translation,cutout', channels_first=False )) - 1 ) ** 2)

        logger.add_scalar('G/YX/fake', g_loss, iteration)

        if opts.use_cycle_consistency_loss:
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.mean((images_Y - G_XtoY(G_YtoX(images_Y)))**2)

            g_loss += opts.lambda_cycle * cycle_consistency_loss
            logger.add_scalar('G/YX/cycle', cycle_consistency_loss, iteration)

        # backprop the aggregated g losses and update G_XtoY and G_YtoX
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print(
                'Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | '
                'd_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    iteration, opts.train_iters, d_real_loss.item(),
                    D_Y_loss.item(), D_X_loss.item(),
                    d_fake_loss.item(), g_loss.item()
                )

            )
            # wandb.log({'D/real_loss': d_real_loss.item(), 'D/Y_loss': D_Y_loss.item(),'D/X_loss': D_X_loss.item(),'D/fake_loss': d_fake_loss.item(),'G/loss': g_loss.item()})


        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)


def main(opts):
    """Loads the data and starts the training loop."""
    # Create  dataloaders for images from the two domains X and Y
    dataloader_X = get_data_loader(opts.X, opts=opts)
    dataloader_Y = get_data_loader(opts.Y, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_Y, opts)
    # wandb.finish()



def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--disc', type=str, default='dc')  # or 'patch'
    parser.add_argument('--gen', type=str, default='cycle')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--use_cycle_consistency_loss', action='store_true')
    parser.add_argument('--init_zero_weights', action='store_true')
    parser.add_argument('--init_type', type=str, default='naive')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lambda_cycle', type=float, default=10)

    # Data sources
    parser.add_argument('--X', type=str, default='cat/grumpifyAprocessed')
    parser.add_argument('--Y', type=str, default='cat/grumpifyBprocessed')
    parser.add_argument('--ext', type=str, default='*.png')
    parser.add_argument('--use_diffaug', action='store_true')
    parser.add_argument('--data_preprocess', type=str, default='deluxe')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='cyclegan')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--checkpoint_every', type=int, default=800)

    parser.add_argument('--gpu', type=str, default='0')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu
    opts.sample_dir = os.path.join(
        'output/', opts.sample_dir,
        '%s_%g' % (opts.X.split('/')[0], opts.lambda_cycle)
    )
    opts.sample_dir += '%s_%s_%s_%s_%s' % (
        opts.data_preprocess, opts.norm, opts.disc, opts.gen, opts.init_type
    )
    if opts.use_cycle_consistency_loss:
        opts.sample_dir += '_cycle'
    if opts.use_diffaug:
        opts.sample_dir += '_diffaug'

    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)

    logger = SummaryWriter(opts.sample_dir)

    print_opts(opts)
    main(opts)
