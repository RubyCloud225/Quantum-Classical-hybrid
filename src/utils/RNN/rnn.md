# Echo RNN Pipeline Documentation

## Architecture Overview

The Echo RNN pipeline is designed to process sequential data using a recurrent neural network architecture that incorporates feedback loops to maintain and update hidden states over time. The main components include:

- Input Layer: Receives the current input vector.
- RNN Cell: Processes the input along with the previous hidden state.
- Feedback Loop: Feeds the hidden state back into the RNN cell for the next timestep.
- Output Layer: Produces the output based on the current hidden state.

```
+------------+       +------------+       +------------+
|  Input x_t | ----> |  RNN Cell  | ----> |  Output y_t|
+------------+       +------------+       +------------+
                         ^   |
                         |   v
                  +----------------+
                  | Hidden State h |
                  +----------------+
```

## RNN Cell (Unitary Pulse)

The core of the Echo RNN is the RNN cell which applies a unitary pulse update to the hidden state. The update rule is given by:

\[
h_t = \sigma(W x_t + U h_{t-1} + b)
\]

where:

- \(h_t\) is the hidden state at time \(t\),
- \(x_t\) is the input at time \(t\),
- \(W\) is the input weight matrix,
- \(U\) is the recurrent weight matrix,
- \(b\) is the bias vector,
- \(\sigma\) is the activation function (e.g., tanh or ReLU).

The unitary pulse ensures stability and efficient gradient flow through time.

## Pipeline with Feedback

The pipeline can be visualized as follows:

```
Time t-1:       Time t:         Time t+1:
+---------+     +---------+     +---------+
| h_{t-1} | --> | RNN Cell| --> | h_t     |
+---------+     +---------+     +---------+
     ^               |               |
     |               v               v
     +----------- Feedback Loop -----------+
```

At each timestep, the hidden state from the previous timestep is fed back into the RNN cell along with the current input.

## Implementation Strategy

The Echo RNN pipeline benefits from GPU acceleration using CUDA kernels to parallelize the matrix operations involved in the RNN cell.

Example CUDA kernel for the RNN cell update:

```cpp
__global__ void rnn_cell_update(
    const float* input,       // Input vector x_t
    const float* hidden_prev, // Previous hidden state h_{t-1}
    const float* W,           // Input weights
    const float* U,           // Recurrent weights
    const float* b,           // Bias
    float* hidden_next,       // Output hidden state h_t
    int input_dim,
    int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        float sum = b[idx];
        // Compute W * x_t
        for (int i = 0; i < input_dim; ++i) {
            sum += W[idx * input_dim + i] * input[i];
        }
        // Compute U * h_{t-1}
        for (int j = 0; j < hidden_dim; ++j) {
            sum += U[idx * hidden_dim + j] * hidden_prev[j];
        }
        // Apply activation function (tanh)
        hidden_next[idx] = tanhf(sum);
    }
}
```

This kernel computes the next hidden state in parallel for all hidden units.

## Next Steps

- **Reinjection of Outputs:** Explore different reinjection strategies where the output \(y_t\) is fed back into the network or combined with the hidden state for richer dynamics.
- **Hidden Dimension Size:** Experiment with varying the hidden dimension size to balance model capacity and computational efficiency.
- **Optimization:** Implement gradient computation and backpropagation through time (BPTT) for training.
- **Integration:** Incorporate the Echo RNN pipeline into larger quantum-classical hybrid models for sequence processing tasks.

## Literature Review: Unitary States & RNN

### "Unitary Evolution Recurrent Neural Networks"  
Arjovsky, M., Shah, A., & Bengio, Y. (2016)  
[https://arxiv.org/abs/1511.06464](https://arxiv.org/abs/1511.06464)  

This paper introduces the concept of unitary recurrent neural networks where the recurrent weight matrix \(U\) is constrained to be unitary (i.e., \(U^\dagger U = I\)), preserving the norm of the hidden state vector through time. The key update is:  
\[
h_t = \sigma(U h_{t-1} + W x_t + b)
\]  
with \(U\) unitary. This constraint addresses the vanishing and exploding gradient problems by ensuring stable gradient norms during backpropagation through time. The authors propose parameterizations of unitary matrices that allow efficient optimization.

**Why it matters:** Stability and long-term dependency modeling are critical for RNNs, and unitary constraints provide a mathematically principled solution.

---

### "Full-Capacity Unitary Recurrent Neural Networks"  
Wisdom, S., Powers, T., Hershey, J. R., Le Roux, J., & Atlas, L. (2016)  
[https://arxiv.org/abs/1611.02745](https://arxiv.org/abs/1611.02745)  

This work extends unitary RNNs by proposing a parameterization that allows the recurrent matrix to be any unitary matrix (full capacity), rather than restricted subsets. The authors use a product of simpler unitary matrices (e.g., diagonal, reflection, Fourier transform matrices) to represent \(U\). The update remains:  
\[
h_t = \sigma(U h_{t-1} + W x_t + b)
\]  
with \(U\) full-capacity unitary.

**Why it matters:** Allows more expressive power while maintaining the benefits of unitary matrices for gradient stability.

---

### "Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections"  
Jing, L., Shen, Y., Dubcek, T., Peurifoy, J., Skirlo, S., LeCun, Y., Tegmark, M., & Soljačić, M. (2017)  
[https://arxiv.org/abs/1705.01691](https://arxiv.org/abs/1705.01691)  

The authors propose using Householder reflections to parametrize orthogonal recurrent matrices \(U\) (i.e., \(U^T U = I\)) efficiently. This method guarantees orthogonality and is computationally efficient. The update is:  
\[
h_t = \sigma(U h_{t-1} + W x_t + b)
\]  
with \(U\) orthogonal.

**Why it matters:** Orthogonal matrices preserve norm and are easier to optimize than general unitary matrices, offering a practical compromise.

---

### "Orthogonal Recurrent Neural Networks with Scaled Cayley Transform"  
Maduranga, D. S., et al. (2019)  
[https://arxiv.org/abs/1905.01040](https://arxiv.org/abs/1905.01040)  

This paper introduces a parametrization of orthogonal matrices using the scaled Cayley transform, allowing for stable and efficient training of orthogonal RNNs. The update rule is similar:  
\[
h_t = \sigma(U h_{t-1} + W x_t + b)
\]  
with \(U\) orthogonal via Cayley transform.

**Why it matters:** Provides an alternative stable parametrization for orthogonal RNNs, improving convergence and performance.

---

### "GORU: Gated Orthogonal Recurrent Unit"  
Jing, L., et al. (2017)  
[https://arxiv.org/abs/1709.01496](https://arxiv.org/abs/1709.01496)  

GORU integrates gating mechanisms from GRUs with orthogonal recurrent matrices to combine stability and gating benefits. The update involves:  
\[
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \sigma(U h_{t-1} + W x_t + b)
\]  
with \(U\) orthogonal and \(z_t\) a gate.

**Why it matters:** Combines the advantages of unitary/orthogonal matrices with gating for better sequence modeling.

---

### "Efficient Unitary Neural Networks using the FFT"  
Emami, A., et al. (2019)  
[https://arxiv.org/abs/1907.02837](https://arxiv.org/abs/1907.02837)  

This work leverages the Fast Fourier Transform (FFT) to efficiently parameterize unitary matrices in RNNs, improving computational efficiency while maintaining unitary constraints.

**Why it matters:** FFT-based parametrization enables scalable and fast unitary RNN training.

---

### Additional Applications and Insights

Unitary and orthogonal RNNs have been applied successfully in speech recognition, natural language processing, and time series prediction, demonstrating improved gradient flow and long-term dependency learning. The mathematical constraints on recurrent matrices prevent exploding/vanishing gradients and improve generalization.

---

### Takeaways for the Echo RNN Pipeline

- **Stability:** Enforcing unitary or orthogonal constraints on the recurrent matrix \(U\) in the Echo RNN ensures stable hidden state norms and gradient flow, mitigating vanishing/exploding gradients.
- **Parametrization:** Efficient parametrizations such as products of simpler unitary matrices, Householder reflections, or Cayley transforms are practical methods to implement these constraints.
- **Expressivity vs. Efficiency:** Full-capacity unitary matrices provide expressivity but may be computationally intensive; orthogonal approximations offer a good trade-off.
- **Integration with Gating:** Incorporating gating mechanisms (e.g., GORU) can further enhance modeling capacity and training stability.
- **GPU Acceleration:** The Echo RNN’s CUDA kernel implementation can be adapted to support these parametrizations and constraints for efficient training.
- **Research Directions:** Exploring reinjection strategies and hidden dimension tuning in the Echo RNN can benefit from these unitary RNN insights to design robust and efficient sequence models.

# Wave Coupling and Anti-Wave Mirrors in RNNs

## Introduction: Coupling of Wave Dimensions in the Echo RNN

The Echo RNN architecture can be extended to process information not just along a single dimension but across multiple coupled wave dimensions. In this context, "wave coupling" refers to the interaction between different hidden state channels or spatial/temporal dimensions, allowing richer dynamics and signal propagation akin to coupled oscillatory systems. This is particularly relevant for tasks where hidden states represent physical or abstract wave phenomena, and their interactions are crucial for learning complex temporal dependencies.

## Mathematical Description: Coupled Cayley Transform

To achieve stable and expressive coupling, the recurrent weight matrix \(U\) is parameterized using a generalized Cayley transform:

\[
U = (I - A)^{-1}(I + A)D
\]

where:
- \(I\) is the identity matrix,
- \(A\) is a skew-Hermitian matrix (\(A^\dagger = -A\)), which can be:
  - Tridiagonal (local coupling between adjacent states),
  - Block skew-Hermitian (coupling in blocks/subspaces),
  - Dense skew-Hermitian (full coupling),
- \(D\) is a diagonal unitary matrix (e.g., phases on the diagonal).

This construction ensures that \(U\) is unitary (or orthogonal in the real case), preserving the norm of the hidden state and allowing stable propagation of coupled waves across the hidden channels.

## Anti-Wave Mirror as a Reflection Operator

The "anti-wave mirror" is an operator \(M\) that acts as a generalized reflection, introducing phase inversion. Mathematically, the mirror operator can be defined as:

\[
M h = R h
\]

where \(R\) is a reflection matrix (e.g., a permutation matrix that reverses the order of components, or a Householder reflection), potentially combined with a diagonal matrix of phase factors (e.g., multiplying selected components by \(-1\) or \(e^{i\pi}\)) to invert the phase:

\[
R = P \cdot \mathrm{diag}(1, \ldots, -1, \ldots, 1)
\]

This operator can model anti-wave boundary conditions or symmetry constraints in the hidden space.

## Gated RNN Equations Including the Mirror

Extending the gated RNN (e.g., GORU) to include the anti-wave mirror, the update equations become:

\[
\tilde{h}_t = \sigma(U h_{t-1} + W x_t + b), \quad
h_t = z_t \odot h_{t-1} + (1-z_t) \odot (M \tilde{h}_t)
\]

where:
- \(\tilde{h}_t\) is the candidate hidden state,
- \(U\) is the coupled unitary matrix via the Cayley transform,
- \(W\) is the input weight matrix,
- \(b\) is the bias,
- \(\sigma\) is the activation function,
- \(z_t\) is the update gate (\(z_t \in [0,1]\)),
- \(M\) is the anti-wave mirror operator.

This structure allows part of the hidden state to be reflected with phase inversion, enabling richer wave interference and symmetry effects in the RNN dynamics.

## CUDA Kernel Sketch: Applying the Anti-Wave Mirror

Below is a high-level sketch of a CUDA kernel that applies the anti-wave mirror after the Cayley transform and nonlinearity. The comments explain each step.

```cpp
// CUDA kernel for gated Echo RNN with coupled Cayley transform and anti-wave mirror
__global__ void echo_rnn_wave_coupling(
    const float* x_t,          // Input vector
    const float* h_prev,       // Previous hidden state
    const float* W,            // Input weight matrix
    const float* A,            // Skew-Hermitian matrix for Cayley transform
    const float* D,            // Diagonal phase matrix
    const float* b,            // Bias
    const float* z_t,          // Update gate
    float* h_next,             // Output hidden state
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        // Step 1: Cayley transform to compute U * h_prev
        // (Assume helper functions for matrix-vector ops and Cayley inversion)
        float Uh_prev = cayley_transform(A, D, h_prev, idx, hidden_dim);

        // Step 2: Compute W * x_t + bias
        float Wx = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            Wx += W[idx * hidden_dim + i] * x_t[i];
        }
        float preact = Uh_prev + Wx + b[idx];

        // Step 3: Apply activation function (e.g., tanh)
        float h_tilde = tanhf(preact);

        // Step 4: Apply anti-wave mirror (reflection with possible phase inversion)
        // For example, reflect index: mirrored_idx = hidden_dim - 1 - idx
        float mirrored = h_tilde;
        if (/* idx is in anti-phase region */) {
            mirrored = -h_tilde; // Phase inversion
        }
        // Optionally, assign to mirrored_idx if spatial reflection is needed

        // Step 5: Gated update
        h_next[idx] = z_t[idx] * h_prev[idx] + (1.0f - z_t[idx]) * mirrored;
    }
}
```

**Comments:**
- Step 1: Computes the action of the coupled unitary matrix \(U\) on the previous hidden state using the Cayley transform.
- Step 2: Computes the input projection and adds bias.
- Step 3: Applies the activation function (e.g., tanh).
- Step 4: Applies the anti-wave mirror, which may include spatial reflection and/or phase inversion depending on the index.
- Step 5: Performs the gated update, blending the previous state and mirrored candidate state.

This kernel can be adapted for complex-valued or block-structured hidden states, and for more general forms of the mirror operator.