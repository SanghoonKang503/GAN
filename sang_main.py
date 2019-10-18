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
from torchvision import datasets, transforms
from torch.autograd import Variable

from sang_gan import *
from sang_plot import *
from sang_utils import *


# training parameters
parser=argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help = "Running Iterations")
parser.add_argument("--latent_dim", type=int, default=100, help = "Latent dimension z")
parser.add_argument("--batch_size", type=int, default=64, help= "Size of the Batches")
parser.add_argument("--lr", type=int, default=0.0001, help="Adam Learning rate")
parser.add_argument("--b1", type=int, default=0.5, help="Momentum of Adam beta1")
parser.add_argument("--b2", type=int, default=0.999, help="Momentum of Adam beta2")
parser.add_argument("--img_size", type=int, default=64, help="Size of input Image")
parser.add_argument("--n_critic", type=int, default=5, help="Number of training step of Discriminator")
parser.add_argument("--lambda", type=int, default=10, help="Lambda of Gradient Descent")

opt= parser.parse_args()

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
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# results save folder
os.mkdir('CelebA_WGAN_results_3', exist_ok=True)
os.mkdir('CelebA_WGAN_results_3/Random_results', exist_ok=True)
os.mkdir('CelebA_WGAN_results_3/Fixed_results', exist_ok=True)

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
    Wassestein_Distance = []

    epoch_start_time = time.time()
    for i, (x_, _) in enumerate(train_loader):
        # Configure input
        x = Variable(x_.cuda())

        # train discriminator D
        D.zero_grad()

        mini_batch = x.shape[0]                                 # image shape
        z = Variable(torch.randn((mini_batch, 100)).view(-1, 100, 1, 1))  # declare noise z = (image_shape, 100, 1, 1)
        z = Variable(z.cuda())

        

        D_real_loss = D(x_)
        D_real_loss = D_real_loss.mean(0).view(1)
        D_real_loss.backward(one)

        fake_image = G(z)
        D_fake_loss = D(fake_image)
        D_fake_loss = D_fake_loss.mean(0).view(1)
        D_fake_loss.backward(mone)

        D_loss=D_fake_loss-D_real_loss
        Wasserstein_D = D_real_loss - D_fake_loss


        # Discriminator Loss

        D_optimizer.step()
        D_losses.append(D_loss.item())

        for p in D.parameters():
            p.data.clamp_(-0.01,0.01)

        if i % 5 == 0:
            for p in D.parameters():
                p.requires_grad = False

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())

            fake_image = G(z_)

            G_loss=D(fake_image)
            G_loss=G_loss.mean().mean(0).view(1)
            G_loss.backward(one)

            G_optimizer.step()

            G_losses.append(G_loss.item())

            Wassestein_Distance.append(Wasserstein_D.item())

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - epoch time: %.2f, loss_d: %.3f, loss_g: %.3f, Wasserstein length: %.3f'
          % ((epoch + 1), opt.n_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
    torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(Wassestein_Distance))))

    p = 'CelebA_WGAN_results_3/Random_results/CelebA_WGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'CelebA_WGAN_results_3/Fixed_results/CelebA_WGAN_' + str(epoch + 1) + '.png'

    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())
    show_result((epoch + 1), z_, save=True, path=p)
    show_result((epoch + 1), fixed_z_, save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.n_epochs, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "CelebA_WGAN_results_3/generator_param.pkl")
torch.save(D.state_dict(), "CelebA_WGAN_results_3/discriminator_param.pkl")
with open('CelebA_WGAN_results_3/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='CelebA_WGAN_results_3/CelebA_WGAN_train_hist.png')

images = []
for e in range(opt.n_epochs):
    img_name = 'CelebA_WGAN_results_3/Fixed_results/CelebA_WGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('CelebA_WGAN_results_3/generation_animation.gif', images, fps=5)
