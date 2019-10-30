import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import autograd

# G(z)
# Generate fake image
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, 128, 4, 1, 0)
        self.deconv2 = nn.ConvTranspose2d(128, 256, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 512, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(512, 1024, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(1024)
        self.deconv5 = nn.ConvTranspose2d(1024, 3, 4, 2, 1)

    # weight_init
    def weight_init(self):
        for m in self._modules:
            #normal_init(self._modules[m], mean, std)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.xavier_normal_(m.bias)

    # forward method
    def forward(self, inputs):
        x = F.relu(self.deconv1(inputs))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 1024, 4, 2, 1)
        self.conv2 = nn.Conv2d(1024, 512, 4, 2, 1)
        self.conv3 = nn.Conv2d(512, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 128, 4, 2, 1)
        self.conv5 = nn.Conv2d(128, 1, 4, 1, 0)


    # weight_init
    def weight_init(self):
        for m in self._modules:
            #normal_init(self._modules[m], mean, std)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.xavier_normal_(m.bias)

    # forward method
    def forward(self, inputs):
        x = F.leaky_relu(self.conv1(inputs), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)

        return x

# Initailize weight of CNN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


def calculate_gradient_penalty(Discriminator, real_images, fake_images, lambda_gp):
    eta = torch.FloatTensor(real_images.size(0), 1, 1, 1).uniform_(0, 1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)
    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = Discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                              create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return grad_penalty