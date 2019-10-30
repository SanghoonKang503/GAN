import matplotlib.pyplot as plt
import itertools
import imageio
import os
import torch
from torch.autograd import Variable


def show_result(Generator, num_epoch, z_, show=False, save=False, path='result.png'):

    Generator.eval()
    test_images = Generator(z_)
    Generator.train()

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

def make_animation(epochs, paths):
    images=[]
    for e in range(epochs):
        img_name = os.path.join(paths, '/Fixed_results/CelebA_WGAN_' + str(e + 1) + '.png')
        images.append(imageio.imread(img_name))
    imageio.mimsave(paths + '/generation_animation.gif', images, fps=5)