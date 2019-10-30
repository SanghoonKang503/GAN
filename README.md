# WGAN_practices
This is code of WGAN-GP using celebA datasets.

# Explanation of file
```
sang_gan.py    : include model of generator & discriminator and gradient_penalty
sang_main.py   : main file of the code 
sang_plot.py   : functions of plotting generator & discriminator loss, generate images, animation of generate images  
sang_utils.py  : dataloader of CelebA
wrapper.py     : training with parameters
```

# Usage

In 'sang_main.py', Some hyperparameter which can changed by users included in parser. 
> 'latent_dim' : input of latent vector in Generator, normally defaults at 100.   

> 'b1, b2'     : Adam optimizer parameters (beta 1,2). 

> 'img_size'   : training image size.
 
> 'n_critic'   : number of discriminator iteration after that, generator updates one time

> 'lambda_gp'  : gradient descent parameter


Users can change the 'iter_list' to shuffle in function 'wrapper'
> function 'wrapper' runs with the set of 'iter_list' automatically. 

# Generated images
<img src="/WGAN-GP_epoch_100_lr_0.0005_batches_64/Random_results/CelebA_WGAN-GP_93.jpg" width="40%" height="30%" title="ABC" alt="RubberDuck"></img>
