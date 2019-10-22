import torch
import os
import argparse
import itertools
import time

import torch.optim as optim

from sang_utils import *
from sang_gan import *
from sang_plot import *

parser=argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help = "Latent dimension z")
parser.add_argument("--b1", type=int, default=0.5, help="Momentum of Adam beta1")
parser.add_argument("--b2", type=int, default=0.999, help="Momentum of Adam beta2")
parser.add_argument("--img_size", type=int, default=64, help="Size of input Image")
parser.add_argument("--n_critic", type=int, default=5, help="Number of training step of Discriminator")
opt= parser.parse_args()

param = {'num_epochs': [10, 50, 100],
         'learning_rate': [0.0001, 0.0005, 0.00005],
         'batch_size' :[64, 128]
         }

product_set = itertools.product(param['num_epochs'],
                                param['learning_rate'],
                                param['batch_size']
                                )


def wrapper_(param, epochs, lr, batches):
    # for save file name
    epoch_ = param['num_epochs']
    lr_ = param['learning_rate']
    bs_ = param['batch_size']

    data_dir = 'resized_celebA'  # this path depends on your computer
    train_loader = get_train_loader(data_dir, batches, opt.img_size)

    lamda_gp = 10

    G = generator()
    D = discriminator()

    # Weight initialization
    G.weight_init()
    D.weight_init()

    # put G and D in cuda
    G.cuda()
    D.cuda()

    # RMSprop optimizer for WGAN
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    # lr_sche_G = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=2, gamma=0.1)
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(opt.b1, opt.b2))

    save_path = 'WGAN-GP_'+ 'epoch_' + str(epochs) + '_lr_' + str(lr) + '_batches_'+ str(batches)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/Random_results', exist_ok=True)
    os.makedirs(save_path + '/Fixed_results', exist_ok=True)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    # fixed noise
    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())

    fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        fixed_z_ = Variable(fixed_z_.cuda())

    print('Training start!')
    start_time = time.time()
    for epoch in range(epochs):
        D_losses = []
        G_losses = []

        epoch_start_time = time.time()
        for i, (x_, _) in enumerate(train_loader):
            # Configure input
            real_image = Variable(x_.cuda())

            # train discriminator D
            D_optimizer.zero_grad()

            mini_batch = real_image.shape[0]  # image shape
            z = Variable(torch.randn((mini_batch, 100)).view(-1, 100, 1, 1))  # declare noise z = (image_shape, 100, 1, 1)
            z = Variable(z.cuda())

            # Generate fake image
            fake_image = G(z)

            real_validity = D(real_image)
            fake_validity = D(fake_image)

            gradient_penalty = calculate_gradient_penalty(D, real_image, fake_image, lamda_gp)

            D_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

            D_loss.backward()
            D_optimizer.step()
            D_losses.append(D_loss.item())

            G_optimizer.zero_grad()

            if i % opt.n_critic == 0:
                # train generator G
                fake_image = G(z)
                fake_validity = D(fake_image)

                G_loss = -torch.mean(fake_validity)

                G_loss.backward()
                G_optimizer.step()
                # lr_sche_G.step()
                G_losses.append(G_loss.item())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - epoch time: %.2f, loss_d: %.3f, loss_g: %.3f'
              % ((epoch + 1), epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                 torch.mean(torch.FloatTensor(G_losses))))

        p = save_path + '/Random_results/CelebA_WGAN-GP_' + str(epoch + 1) + '.png'
        fixed_p = save_path + '/Fixed_results/CelebA_WGAN-GP_' + str(epoch + 1) + '.png'

        show_result(G, (epoch + 1), z_, save=True, path=p)
        show_result(G, (epoch + 1), fixed_z_, save=True, path=fixed_p)

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f"
          % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), epochs, total_ptime))

    print("Training finish!... save training results")
    show_train_hist(train_hist, save=True, path=save_path + '/CelebA_WGAN-GP_train_hist.png')
    make_animation(epochs, save_path)

for num_epochs, learning_rate, batch_size in product_set:
    wrapper_(param, num_epochs, learning_rate, batch_size)