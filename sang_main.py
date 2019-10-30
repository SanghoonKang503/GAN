import argparse
import itertools

from sang_wrapper import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension z")
    parser.add_argument("--b1", type=int, default=0.5, help="Momentum of Adam beta1")
    parser.add_argument("--b2", type=int, default=0.999, help="Momentum of Adam beta2")
    parser.add_argument("--img_size", type=int, default=64, help="Size of input Image")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of training step of Discriminator")
    parser.add_argument("--lambda_gp", type=int, default=10, help="Lambda for gradient penalty")
    parser.add_argument("--dataset", default='cifar10', help='cifar10, celebA')
    opt = parser.parse_args()
    print(opt)
    param = opt.__dict__

    iter_list = {'num_epochs': [100],
                 'learning_rate': [0.0005, 0.0003],
                 'batch_size': [64]
                 }

    product_set = itertools.product(iter_list['num_epochs'],
                                    iter_list['learning_rate'],
                                    iter_list['batch_size'])

    for num_epochs, learning_rate, batch_size in product_set:
        param['num_epochs'] = num_epochs
        param['learning_rate'] = learning_rate
        param['batch_size'] = batch_size
        wrapper_(param)
