import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from sang_gan import *
from sang_plot import *
from sang_utils import *


# training parameters
lr = 0.00005
train_epoch = 200
img_size = 64

batch_sizes = 64
data_dir = 'resized_celebA'  # this path depends on your computer

# put image data into data_loader
train_loader = get_train_loader(data_dir, batch_sizes, img_size)

# network 선언, () number는 filter 수
G = generator()
D = discriminator()

# Weight initialization
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

# put G&D in cuda
G.cuda()
D.cuda()

# RMSprop optimizer for WGAN
G_optimizer = optim.RMSprop(G.parameters(), lr=lr)
D_optimizer = optim.RMSprop(D.parameters(), lr=lr)

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
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    Wassestein_Distance = []
    num_iter = 0

    epoch_start_time = time.time()
    for i, (x_, _) in enumerate(train_loader):

        for p in D.parameters():
            p.requires_grad=True

        # train discriminator D
        D.zero_grad()

        one = torch.FloatTensor([1])
        mone = one * -1
        one= one.cuda()
        mone = mone.cuda()

        mini_batch = x_.size()[0]

        x_ = Variable(x_.cuda())

        D_real_loss = D(x_)
        D_real_loss = D_real_loss.mean(0).view(1)
        D_real_loss.backward(one)

        # z_를 N(0,1)의 확률분포에서 (mini_batch, 100)의 형태로 선택하고 (100, 1, 1)로 변형
        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())
        fake_image = G(z_)
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

            num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - epoch time: %.2f, loss_d: %.3f, loss_g: %.3f, Wasserstein length: %.3f' % (
    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
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
torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "CelebA_WGAN_results_3/generator_param.pkl")
torch.save(D.state_dict(), "CelebA_WGAN_results_3/discriminator_param.pkl")
with open('CelebA_WGAN_results_3/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='CelebA_WGAN_results_3/CelebA_WGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'CelebA_WGAN_results_3/Fixed_results/CelebA_WGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('CelebA_WGAN_results_3/generation_animation.gif', images, fps=5)
