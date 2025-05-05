https://github.com/facebookresearch/DiT/blob/main/diffusion/diffusion_utils.py

calculate warm up beta 
  params: 
    - beta start
    - beta end
    - num diffusion timestamps
    - type = using float 64
    - warm up - frac

get the beta schedule
    need a beta schedule to sample from the diffusion process
        - beta schedule is a list of beta values
        - using the linear regression for this 
    schedule * beta start, beta end , num diffusion timestamp

get a predefined beta schedule- this has to be similar to the limit in our diffusion timestamp 

create a schedule that discretizes the given alpha_t function
    - alpha_t function is a function that takes in a float and returns a float
    - the schedule is a list of tuples where each tuple contains a float and the corresponding alpha_t values
    - the schedule is created by discretizing the given alpha_t function into num_steps steps
    - the schedule is then sorted by the float values in ascending order
    - the schedule is then returned
    - the schedule is used to create the beta schedule
params:
    - num diffusion timestamps 
    - alpha_t function - using lambda taking the argument t from 0 to 1 and produces a cumulative product of (1-beta)
    - max-beta - the maximum beta value - lower than 1 to prevent singularities.

Above is our params for Diffusion

this is our maindiffusion.cpp
use float 64 for t he beta
betas, model mean type, model var type, loss type

Calculate q(x_t | x_{t-1}, t) - this is the forward process

then posterior q(x_{t-1} | x_t, t) - this is the reverse process

log the calculation of the reverse process

first calculate the distribution of q(x_t | x_{t-1}, t)

apply model to get the p(x_{t-1} | x_t) and predict x, x_0
params: 
    - model takes signal(in this case the noise and loss classes) and batch of timesteps as its input
    - x: the [N x C x ....] at time of t
    - t: a 1 D vector of timesteps
    - clip denoised: clip the denoised signal into [-1, 1]
    - denoised_fn: if none apple x_start preduction before applying the clip denoised
    - model Kwags : model kwags for the model- dict of extra keywords to pass to the model
    - return:
        - mean
        - variance
        - log_variance
        - pred_xstart

predict x_start from eps
predict eps from xstart 

condition the mean by computing the graient of the log proability of the mean with respect to the mean
    computes grad(log(p(y|x)))
    the condition is on y
    strategy used is Sohl- Dickstein et al (2015) - lets look for something more recient 

use the reverse mode automatic differentiation to compute the gradient of the log probability of the mean with respect to the mean
https://arxiv.org/pdf/1509.07164 
- this is the reverse process
f(y,µ,σ) = log (Normal(y|µ,σ)) =−1/2
(y−µ/σ)^2 −log σ−1/2 log(2π)

with a gradient

∂f/∂y(y,µ,σ) =−(y−µ)σ^-2
∂f/∂µ(y,µ,σ) = (y−µ)σ^-2
∂f/∂σ(y,µ,σ) =−(y−µ)^2σ^−3 − σ^−1

condition the score:
    compute the P_mean_variance output

Sample x_{t-1} at given timestep
    params:
        - model: model above
        - x: the current vector at x_{t-1}
        - t: value of t starting at 0 for first diffusion step
        - clip denoised- clip the x_start prediction top [-1, 1]
        - denoised_fn- if none apply x_start prediction before applying the clip denoised
        - cond_fn - gradient function acts similarly to the model
        - model _kwargs = extra keywords for conitioning 
        - return:
            - sample - random
            _ predict X_start - predict x_0

generate a p_sample loop
    params:
        - model: model above
        - shape: shape of sample (N, C, H, W)
        - noise: noise schedule- same as sample
        - clip denoised
        - denoised_fn
        - model _kwargs
        - device: use the models parameters
        - progress - show a tqdm progress bar
    return:
        non differentiable batch of samples

Generate a P_sample _loop_progressibe
    params: 
    - yield intermediate samples from each timestamp of diffusion
    - arguments are same as P_sample Loop
    - returns:
        - generator over dicts where each one returns p_sample()


    

