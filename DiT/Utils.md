Influence taken from https://github.com/facebookresearch/DiT/blob/main/diffusion/diffusion_utils.py

Utils = Calculate the KL divergence between two gaussians
shapes are automatically broadcasted - batch compaired to scale

Params: 
    - Mean 1
    - var 1
    - mean 2
    - var 2

for the purposes of this exercise it will be noise of tokens against the loss function of NLL

Next util - approximate the cumulative distribution function of the standard normal - this will be the noise

util 3  - calculate the log likelhold of continuous gassian distribution
    params: 
        - x: the target
        - mean: the Gaussian mean of vector
        - log scales: the log of Stddev
    return: a vector like x of log proabilities (in nats)

util 4 = discretized gaussian log likelhood
    params:
        - x: the target assumed to be uint8 values rescaled to range 
        - means
        - log scales
        return: a vector like x of log probabilities (in nats)

