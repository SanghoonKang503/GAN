import os
import time

import torch.optim as optim

from sang_utils import *
from sang_gan import *
from sang_plot import *
    

def wrapper_(opt):
    # for save file name
    epochs = opt['num_epochs']
    lr = opt['learning_rate']
    bs = opt['batch_size']

    if opt['dataset'] == "celebA":
        data_dir = 'resized_celebA'  # this path depends on your computer
        train_loader = get_celebA_loader(data_dir, bs, opt['img_size'])

    elif opt['dataset'] == "cifar10":
        train_loader = get_cifar10_loader(bs, opt['img_size'])


    G = generator()
    D = discriminator()

    # Weight initialization
    G.weight_init()
    D.weight_init()

    # put G and D in cuda
    G.cuda()
    D.cuda()

    # Adam optimizer for WGAN-GP
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(opt['b1'], opt['b2']))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(opt['b1'], opt['b2']))
    # lr_sche_G = torch.optim.lr_scheduler.StepLR(G_optimizer, step_size=10, gamma=0.99)
    # lr_sche_D = torch.optim.lr_scheduler.StepLR(D_optimizer, step_size=10, gamma=0.99)

    save_path = f'Cifar10_WGAN-GP_epoch_{epochs}_lr_{lr}_batches_{bs}'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'Random_results'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'Fixed_results'), exist_ok=True)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['total_ptime'] = []

    # fixed noise
    fixed_z_ = torch.randn((5 * 5, opt['latent_dim'])).view(-1, opt['latent_dim'], 1, 1)
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
            z = Variable(torch.randn((mini_batch, opt['latent_dim'])).view(-1, opt['latent_dim'], 1, 1))  # declare noise z = (image_shape, 100, 1, 1)
            z = Variable(z.cuda())

            # Generate fake image
            fake_image = G(z)

            real_validity = D(real_image)
            fake_validity = D(fake_image)

            gradient_penalty = calculate_gradient_penalty(D, real_image, fake_image, opt['lambda_gp'])

            D_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

            D_loss.backward()
            D_optimizer.step()
            # lr_sche_D.step()
            D_losses.append(D_loss.item())

            G_optimizer.zero_grad()

            if i % opt['n_critic'] == 0:
                # train generator G
                fake_image = G(z)
                fake_validity = D(fake_image)

                G_loss = -torch.mean(fake_validity)

                G_loss.backward()
                G_optimizer.step()
                # lr_sche_G.step()
                G_losses.append(G_loss.item())

        # For random generator images
        z_ = torch.randn((5 * 5, opt['latent_dim'])).view(-1, opt['latent_dim'], 1, 1)
        with torch.no_grad():
            z_ = Variable(z_.cuda())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print(f'[{epoch+1}/{epochs}] - epoch time: {per_epoch_ptime}, loss_D: {torch.mean(torch.FloatTensor(D_losses))}, loss_G: {torch.mean(torch.FloatTensor(G_losses))}')

        p = os.path.join(save_path, 'Random_results/CelebA_WGAN-GP_'+str(epoch+1)+'.png')
        fixed_p = os.path.join(save_path, 'Fixed_results/CelebA_WGAN-GP_'+str(epoch+1)+'.png')

        show_result(G, (epoch + 1), z_, save=True, path=p)
        show_result(G, (epoch + 1), fixed_z_, save=True, path=fixed_p)

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print(f'Total time: {total_ptime}')
    print("Training finish!... save training results")
    show_train_hist(train_hist, save=True, path=save_path + '/CelebA_WGAN-GP_train_hist.png')
    make_animation(epoch, save_path)