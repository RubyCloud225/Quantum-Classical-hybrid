** HyperNetwork - simulation - influenced by Learning to forget using Hypernetworks- Rangel et all 2025

Dimensions - 1536
Hidden Layers - 12
Attention Heads - 16 use Adam W optimizer with learning rate 0.0001

Latent Diffusion Transformer 

need to embed and calculate noise using Gaussain noise

    Noise       sum
    | |          |
    | |  Linear and Reshape   |
    | |  Layer Norm           |
    | |  Diffusion Process    |
    | |  Patchify, Embed      |
    | |  Noised Latient, timestamp (t), label (l)  |


DiT block

            MLP -> Conditioning 
                |
                V
        |   Scale a2                |
        |   Pointwise Feedforward   |
        |   Scale, Shift Y2, B2     |
        |   LayerNorm               |
        |   Scale a1                |
        |   MultiHead Attention     |
        |   Scale, Shift Y1, B1     |
        |   LayerNorm               |
        |   Input Tokens            |

AFter which decode -> Predict 0*2

Tokenization -> Embedding -> Data