import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.autograd as autograd

from torchvision import datasets, transforms
from torch.autograd import Variable

from sang_gan import *
from sang_plot import *
from sang_utils import *


# training parameters
parser=argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10, help = "Running Iterations")
parser.add_argument("--latent_dim", type=int, default=100, help = "Latent dimension z")
parser.add_argument("--batch_size", type=int, default=128, help= "Size of the Batches")
parser.add_argument("--lr", type=int, default=0.0001, help="Adam Learning rate")
parser.add_argument("--b1", type=int, default=0.5, help="Momentum of Adam beta1")
parser.add_argument("--b2", type=int, default=0.999, help="Momentum of Adam beta2")
parser.add_argument("--img_size", type=int, default=64, help="Size of input Image")
parser.add_argument("--n_critic", type=int, default=5, help="Number of training step of Discriminator")

opt= parser.parse_args()
lamda_gp = 10
# put image data into data_loader
data_dir = 'resized_celebA'  # this path depends on your computer
train_loader = get_train_loader(data_dir, opt.batch_size, opt.img_size)

# Declare G and D network
G = generator()
D = discriminator()

# Weight initialization
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# put G and D in cuda
G.cuda()
D.cuda()

# RMSprop optimizer for WGAN
G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
lr_sche_G = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=2, gamma=0.1)

D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# results save folder
os.makedirs('CelebA_WGAN_results_0', exist_ok=True)
os.makedirs('CelebA_WGAN_results_0/Random_results', exist_ok=True)
os.makedirs('CelebA_WGAN_results_0/Fixed_results', exist_ok=True)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)  # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())

print('Training start!')
start_time = time.time()
for epoch in range(opt.n_epochs):
    D_losses = []
    G_losses = []

    epoch_start_time = time.time()
    for i, (x_, _) in enumerate(train_loader):
        # Configure input
        real_image = Variable(x_.cuda())

        # train discriminator D
        D_optimizer.zero_grad()

        mini_batch = real_image.shape[0]                                           # image shape
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
            #G_optimizer.step()
            lr_sche_G.step()
            G_losses.append(G_loss.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - epoch time: %.2f, loss_d: %.3f, loss_g: %.3f'
          % ((epoch + 1), opt.n_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses))))

    p = 'CelebA_WGAN_results_0/Random_results/CelebA_WGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'CelebA_WGAN_results_0/Fixed_results/CelebA_WGAN_' + str(epoch + 1) + '.png'

    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())
    show_result(G, (epoch + 1), z_, save=True, path=p)
    show_result(G, (epoch + 1), fixed_z_, save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.n_epochs, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "CelebA_WGAN_results_0/generator_param.pkl")
torch.save(D.state_dict(), "CelebA_WGAN_results_0/discriminator_param.pkl")
with open('CelebA_WGAN_results_0/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='CelebA_WGAN_results_0/CelebA_WGAN_train_hist.png')

images = []
for e in range(opt.n_epochs):
    img_name = 'CelebA_WGAN_results_0/Fixed_results/CelebA_WGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('CelebA_WGAN_results_0/generation_animation.gif', images, fps=5)
