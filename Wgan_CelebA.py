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


# G(z)
# Generate fake image
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, 1024, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 4, 1, 0)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x

# Initailize weight of CNN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)  # fixed noise
with torch.no_grad():
    fixed_z_ = Variable(fixed_z_.cuda())


def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    with torch.no_grad():
        z_ = Variable(z_.cuda())

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5 * 5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# training parameters
batch_sizes = 64
lr = 0.00005
train_epoch = 200

img_size = 64

# put image data into data_loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_dir = 'resized_celebA'  # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_sizes, shuffle=True)

# confrimed input image size!
temp = plt.imread(train_loader.dataset.imgs[0][0])
if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_preprocess.py\" !!!')
    sys.exit(1)

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
if not os.path.isdir('CelebA_WGAN_results_3'):
    os.mkdir('CelebA_WGAN_results_3')
if not os.path.isdir('CelebA_WGAN_results_3/Random_results'):
    os.mkdir('CelebA_WGAN_results_3/Random_results')
if not os.path.isdir('CelebA_WGAN_results_3/Fixed_results'):
    os.mkdir('CelebA_WGAN_results_3/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    Wassestein_Distance = []
    num_iter = 0

    # learning rate decay
    """    if (epoch + 1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")"""

    if (epoch + 1) == 101:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

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

    show_result((epoch + 1), save=True, path=p, isFix=False)
    show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
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