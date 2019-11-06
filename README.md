# WGAN_practices
This is code of WGAN-GP using celebA and CIFAR10 datasets.

# Explanation of file
```
sang_gan.py    : includes model of generator & discriminator and gradient_penalty
sang_main.py   : main file of the code 
sang_plot.py   : functions of plotting generator & discriminator loss, generate images, animation of generate images  
sang_utils.py  : dataloader of CelebA & Cifar10
wrapper.py     : training with parameters
```

# Requirments
1. Python(3.6.9)
2. pytorch (1.1.0)
3. matplotlib(3.1.0)
4. numpy (1.16.4)

# Usage

In 'sang_main.py', Some hyperparameter which can changed by users included in parser. 

1. 'dataset'    : select the dataset between Cifar10 and CelebA.   
2. 'latent_dim' : input of latent vector in Generator, normally defaults at 100.   
3. 'b1, b2'     : Adam optimizer parameters (beta 1,2). 
4. 'img_size'   : training image size.
5. 'n_critic'   : number of discriminator iteration after that, generator updates one time
6. 'lambda_gp'  : gradient descent parameter


User runs the 'sang_main.py' with changing the list in 'iter_list' 
-> function 'wrapper' runs with the set of 'iter_list' automatically. 

# Generated images
![CelebA_WGAN-GP_93](https://user-images.githubusercontent.com/33616377/67831745-a22f9d00-fb22-11e9-90de-10b3d11a9a45.png)

