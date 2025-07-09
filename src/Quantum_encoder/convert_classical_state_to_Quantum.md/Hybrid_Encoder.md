The Goal is to turn classical tokens to quantum data. 

This method simultaneously uses Amplitude Encoding to give  real valued
into quantum state amplitudes

then angle encoding by mapping the data to rotation angles for single qubit
gates.

we take our tokens:

    \mathbf{x} = [x_0, x_1, \ldots, x_{n-1}] \in \mathbb{R}^n

encode this to a quantum state 

        |\psi\rangle = \sum_{i=0}^{n-1} \alpha_i \cdot R_{\text{axis}
        (\theta_i) |0\rangle_i}

	\alpha_i = \frac{x_i}{\|\mathbf{x}\|_2} 

are normalized amplitudes
    
    |theta_i = \alpha_i . \pi are rotation angles
    
    \{R}_axis (\theta_i) \mathbb {R_x^1, R_y^1, R_z} is a single qubit gate

We normalise the vector and produce a valid quantum state vector

    \alpha_i = \frac{x_i}{\|\mathbf{x}\_2} = \|\mathbf{x}\|_2 = \sum
    {i = 0^n-1 X^2_i}

    \psi\rangle = \sum_{i=0}^{n-1}\alpha_i \psi\i

## Angle Encoding

each normalised value 

    \alpha_i \mathbb[-1,1]

this is mapped to a gate - rotation angle

    \theta_i = \alpha-i \cdot \pi

to apply this to a qubit 

    R_y(\theta_i) = \cos(\theta_i/2)|0\rangle + \sin(\theta_i/2)|1\rangle

this is to turn the classical state to a quantum state 
to the following

    |\psi\rangle = \bigotimes_{i=0}^{n-1} \alpha_i \cdot R_y(\theta_i) |0\rangle




