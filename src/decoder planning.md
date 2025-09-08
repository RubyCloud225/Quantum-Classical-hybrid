# Skyrme Decoder Layer — Math + Pseudocode

## Overview (short)

Input: an image-like encoding of photons (e.g. polarization-resolved images, or CNN feature map reshaped to a 2D field).
Goal: map that input to a smooth Bloch-field $\mathbf{n}(x,y)\in S^2$ by minimizing an optical baby-Skyrme energy, then extract compact latent descriptors (topological charge, center, size, orientation) to pass to a downstream diffusion/translation model.

---

## 1. Continuous energy functional (baby-Skyrme style)

Define $\mathbf{n}(x,y)=(n_1,n_2,n_3)$ with constraint $|\mathbf{n}|=1$. The effective optical energy functional is:

$$
E[\mathbf{n}] \;=\; \int_\Omega \left[ \frac{A}{2}\,(\nabla \mathbf{n})^2 \;+\; \frac{B}{4}\,\big(\mathbf{n}\cdot(\partial_x\mathbf{n}\times\partial_y\mathbf{n})\big)^2 \;+\; V(\mathbf{n};\mathbf{I}) \right] d^2r,
$$

where:

* $(\nabla\mathbf{n})^2 = \sum_{i=1}^3 \big((\partial_x n_i)^2 + (\partial_y n_i)^2\big)$,
* $q(\mathbf{r}) \equiv \mathbf{n}\cdot(\partial_x\mathbf{n}\times\partial_y\mathbf{n})$ (topological density),
* $V(\mathbf{n};\mathbf{I})$ is a data-dependent potential that anchors $\mathbf{n}$ to the CNN encoder output $\mathbf{I}(x,y)$ (see below),
* $A,B>0$ are coefficients (gradient stiffness and Skyrme stabilizer).

Boundary domain $\Omega$ is the 2D transverse grid.

### Euler–Lagrange gradient flow (for minimisation)

The gradient descent / relaxation PDE (with projection to unit length) is

$$
\frac{\partial \mathbf{n}}{\partial t} = -\frac{\delta E}{\delta \mathbf{n}} + \lambda(\mathbf{r},t)\,\mathbf{n},
$$

where $\lambda$ enforces $|\mathbf{n}|=1$. Explicit variational derivative (informal):

$$
\frac{\delta E}{\delta \mathbf{n}} \approx -A\,\Delta\mathbf{n} - B\,\nabla\!\cdot\!\big(q(\mathbf{r})\,(\partial_y\mathbf{n}\times\hat{x} - \partial_x\mathbf{n}\times\hat{y})\big) + \frac{\partial V}{\partial\mathbf{n}}.
$$

In practice we discretise and implement an iterative update.

---

## 2. Discrete grid & operators

Let the grid be $i=1\ldots N_x$, $j=1\ldots N_y$ with spacing $\Delta x, \Delta y$. Represent $\mathbf{n}_{i,j}$ as a 3-vector at each pixel. Finite differences:

* $\partial_x n_{i,j} \approx (n_{i+1,j}-n_{i-1,j})/(2\Delta x)$
* $\partial_y n_{i,j} \approx (n_{i,j+1}-n_{i,j-1})/(2\Delta y)$
* Laplacian: $\Delta n_{i,j} \approx \frac{n_{i+1,j}+n_{i-1,j}+n_{i,j+1}+n_{i,j-1}-4n_{i,j}}{\Delta x^2}$ (use $\Delta x=\Delta y$ for simplicity).

Topological density per pixel:

$$
q_{i,j} \approx \mathbf{n}_{i,j}\cdot\big((\partial_x\mathbf{n})_{i,j}\times(\partial_y\mathbf{n})_{i,j}\big).
$$

Discrete skyrmion number:

$$
N_{\text{sk}} \approx \frac{1}{4\pi}\sum_{i,j} q_{i,j}\,\Delta x\,\Delta y.
$$

---

## 3. CNN + Skyrme Decoder architecture (math + pseudocode)

### Notation

* `CNN(I_raw) -> F` : a CNN encoder mapping input (image / multi-channel field) to a feature map `F` of shape `[C, H, W]`.
* `Proj(F) -> n0` : a small conv head mapping features to an initial (unnormalized) Bloch field `n0` of shape `[3,H,W]`.
* `Normalize(n0) -> n_init` : per-pixel normalization to unit length: `n_init = n0 / |n0|`.
* `SkyrmeRelax(n_init, I_anchor, steps, params) -> n_final` : iterative solver that minimizes E\[n] and returns relaxed field.
* `ExtractLatents(n_final) -> {N_sk, center, size, orientation, q_map}` : extract descriptors.
* `DiffusionModel(latents) -> output` : downstream generative translator.

---

### Pseudocode (markdown fenced block)

```python
# Hyperparameters
A = ...      # gradient stiffness
B = ...      # skyrme stabilizer
alpha = ...  # anchoring weight for V(n;I)
dt = 0.1     # gradient flow timestep
T = 200      # number of relaxation iterations
eps = 1e-8   # numeric stabilizer

# Forward pass
def forward(I_raw):
    # 1) CNN encoder -> feature map
    F = CNN(I_raw)                 # shape [C, H, W]

    # 2) Project to initial Bloch field (unnormalised)
    n0 = Proj(F)                   # shape [3, H, W], floats

    # 3) Normalise to unit length to get initial n
    norm = sqrt(sum(n0**2 over channel=3) + eps)
    n = n0 / norm                  # n has unit length per pixel

    # 4) Optional: create anchoring field I_anchor from raw input or features
    #    Example: map intensity/polarization channels to preferred n_target
    n_target = AnchorMap(I_raw)    # shape [3, H, W] (unit-normalised)

    # 5) Relaxation loop: gradient descent on E[n]
    for t in range(T):
        # compute finite-difference derivatives: grad_x, grad_y, laplacian
        grad_x = finite_diff_x(n)     # shape [3,H,W]
        grad_y = finite_diff_y(n)
        lap = laplacian(n)

        # topological density q = n · (∂x n × ∂y n)
        q = dot(n, cross(grad_x, grad_y))   # shape [H,W]

        # term1: -A * laplacian (driving towards smoothing)
        term1 = -A * lap

        # term2: -B * divergence of ( q * cross derivatives ) 
        # (discrete approximation; implement as finite-diff of q * appropriate cross-terms)
        # For brevity, denote term2 = -B * SkyrmeGradient(n, grad_x, grad_y, q)
        term2 = -B * SkyrmeGradient(n, grad_x, grad_y, q)

        # term3: anchoring potential gradient: + alpha * (n - n_target) (pulls n->n_target)
        # if n_target is undefined, skip this term
        term3 = -alpha * (n_target - n)  # negative gradient of 0.5*alpha||n-n_target||^2

        # total descent direction
        dE_dn = term1 + term2 + term3

        # gradient descent step
        n = n - dt * dE_dn

        # renormalize to unit-length (project back to S^2)
        n = n / sqrt(sum(n**2 over channel=3) + eps)

    # 6) After relaxation, compute outputs:
    q_map = compute_q_map(n)              # q per pixel
    N_sk = (1/(4*pi)) * sum(q_map)*dx*dy  # approximate integer
    center = sum( r * |q_map| ) / sum(|q_map|)   # weighted centroid of skyrmion density
    size = sqrt( sum( |r-center|^2 * |q_map| ) / sum(|q_map|) )
    orientation = compute_orientation(n) # e.g., principal axis from local gradients

    latents = { 'N_sk':N_sk, 'center':center, 'size':size, 
                'orientation':orientation, 'q_map':q_map }

    # 7) Pass latents to diffusion model (conditioning)
    out = DiffusionModel.ConditionOn(latents)

    return out, n, latents
```

---

## 4. Mathematical details for SkyrmeGradient and anchoring

A practical discrete form for the Skyrme gradient (term2) can be written as:

1. compute

$$
(\partial_x\mathbf{n})_{i,j}\, ,\; (\partial_y\mathbf{n})_{i,j}
$$

2. compute

$$
\mathbf{C}_{i,j} \;=\; q_{i,j}\, \big[(\partial_y\mathbf{n})_{i,j}\times \hat{x} - (\partial_x\mathbf{n})_{i,j}\times \hat{y}\big],
$$

3. then the divergence (finite difference):

$$
(\nabla\cdot\mathbf{C})_{i,j} \approx \frac{\mathbf{C}_{i+1,j}-\mathbf{C}_{i-1,j}}{2\Delta x} + \frac{\mathbf{C}_{i,j+1}-\mathbf{C}_{i,j-1}}{2\Delta y}.
$$

Finally

$$
\text{SkyrmeGradient}_{i,j} = \nabla\cdot\mathbf{C}_{i,j}.
$$

This yields a vector field at each pixel that you multiply by $-B$.

Anchoring potential $V(\mathbf{n};\mathbf{I})$ often chosen as simple quadratic:

$$
V(\mathbf{n};\mathbf{I}) = \frac{\alpha}{2}\|\mathbf{n}-\mathbf{n}_{\text{target}}(\mathbf{I})\|^2,
$$

so $\partial V/\partial\mathbf{n} = \alpha(\mathbf{n}-\mathbf{n}_{\text{target}})$. This ensures the relaxed field remains faithful to encoder-provided information.

---

## 5. Differentiability & end-to-end training

* All steps above are differentiable (finite-difference operations are linear ops). The normalization projection `n = n / |n|` is differentiable almost everywhere (use smooth approx if desired).
* The relaxation loop can be unrolled T steps and trained end-to-end with backprop (treat it like a recurrent layer). Alternatively, treat relaxation as a fixed solver (no gradient) and only train the encoder/projection.
* Loss terms to use during training:

  * `L_anchor = ||n_final - n_target||^2` (fidelity)
  * `L_top = (N_sk - N_target)^2` if you want integer supervision (or cross-entropy if multiple classes)
  * `L_reg = \int (\nabla n)^2` to regularize smoothness
  * downstream task loss (diffusion reconstruction, classification, etc.)

---

## 6. Extracted latents: definitions & formulas

* **Topological charge**

  $$
  N_{\text{sk}} = \frac{1}{4\pi}\,\sum_{i,j} q_{i,j}\,\Delta x\,\Delta y.
  $$
* **Center**

  $$
  \mathbf{r}_c = \frac{\sum_{i,j} \mathbf{r}_{i,j}\,|q_{i,j}|}{\sum_{i,j} |q_{i,j}|}.
  $$
* **Size**

  $$
  R = \sqrt{\frac{\sum | \mathbf{r}_{i,j}-\mathbf{r}_c |^2 \, |q_{i,j}|}{\sum |q_{i,j}|}}.
  $$
* **Orientation**: compute 2×2 inertia tensor of $|q|$ and take principal axis angle.

These latents can be concatenated into a compact vector for conditioning the diffusion model.

---

## 7. Practical implementation notes & hyperparameters

* **Grid resolution**: choose H×W compatible with CNN feature map (e.g., 64×64). Increase to improve skyrmion accuracy.
* **Timestep & iterations**: dt small (0.05–0.2); T 50–500 depending on stiffness. Unroll fewer steps in training, more at inference if heavy smoothing required.
* **Numerical stability**: add eps in normalization; use implicit schemes or smaller dt if blowups occur.
* **Boundary conditions**: periodic or Dirichlet `n = n_target` at edges; periodic simplifies topology calculation.
* **Coefficients**: tune A,B,alpha. Typical regime: A moderate (smoothness), B small but nonzero to stabilize size, alpha sets encoder fidelity.
* **GPU**: all ops are pointwise or small stencils — implement with convolutions (e.g., kernel for finite differences) for speed and autodiff compatibility.
* **Differentiable integrals**: sums are trivially differentiable; do not use argmax-like ops without straight-through estimators.
* **Sparsity & speed**: if q\_map is sparse (localized skyrmions), you can compute latents from bounding boxes for efficiency.

---

## 8. Why this is useful vs a plain CNN alone

* A CNN can map images to features or even directly to latents. The Skyrme decoder *adds physics priors*:

  * forces smoothness and topological consistency,
  * provides an interpretable bottleneck (N\_sk, center, size),
  * regularises the latent manifold so downstream diffusion learns to operate on stable, particle-like objects rather than arbitrary noisy patterns.

Think of the Skyrme module as a **neural+physics layer**: it learns from data (via `Proj` and anchoring) but enforces global topological constraints via energy minimisation.
