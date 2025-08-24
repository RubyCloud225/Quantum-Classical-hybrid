# Wave Propagation Algorithm

Radio wave progagation in plasma reduces to a 2d time - dependent schrodinger equation 

## Core Propagation -> Schrodinger Mapping

$${\psi\}$$ component after factoring out $${\epsilon=1-\omega_p^2/\omega^2 }$$ and using the dielectric function $${\omega\gg\omega_p}$$

core model is 

$${2ik\,\partial_z\psi+\nabla_\perp^2\psi - r_e N(x,z)\,\psi = -\partial_z^2\psi,\qquad r_e=\frac{4\pi e^2}{mc^2}.}$$

removing $${\partial_z^2\psi}$$ gives the schrodinger evolution.

giving rise to: 

$${i\,\partial_z\psi(x,z)=\Big(-\frac{1}{2k}\nabla_\perp^2+\frac{r_e}{2k}N(x,z)\Big)\psi(x,z).\tag{S}}$$

z plays the role of Time, k is like m and the potential is $${V(x,z)=\tfrac{r_e}{2k}N(x,z)}$$ 

### invariants 

the L^2 norm (intensity) is conserved: $${I=\int |\,\psi(x,z)\,|^2\,dx=\text{const}.}$$

define an energy like functional: $${T(z)=\int\!\Big[\frac{1}{2k}|\nabla_\perp\psi|^2+\frac{r_e}{2k}N(x,z)\,|\psi|^2\Big]dx,}$$

with: $${\frac{dT}{dz}=\frac{r_e}{2k}\int\!(\partial_z N)\,|\psi|^2 dx,}$$

so T is constant or adiabatic if N is z-independent or slowly varying ( this would allow for compression of a state which is varying undefinable)

## Liouville/Hamiltonian form

$${\psi=R+iI}$$ is then split into a hamiltonian pair:

$${\partial_z \begin{bmatrix}R\\ I\end{bmatrix} \begin{bmatrix} 0&-\hat D\\ \hat D&0 \end{bmatrix} \begin{bmatrix}R\\ I\end{bmatrix}, \quad \hat D=\frac{1}{2k}\nabla_\perp^2-\frac{r_e}{2k}N(x,z).}$$

generating a incompressible flow in (R,I) phase-space with continuity

$${\partial_z \rho+\nabla_\perp\!\cdot j=0,\quad\rho=|\,\psi\,|^2,\quad j=\frac{1}{k}\,\text{Im}(\psi^\ast\nabla_\perp\psi).}$$

by using the invariants can build a canonical ensemble over fourier modes: 
$${P(\{\psi_q\})\propto \prod_q \exp\!\Big(-\,q\Big[\alpha+\beta\Big(\frac{q^2+r_eN}{2k}\Big)\Big]\,|\psi_q|^2\Big),}$$

over all power:
$${\boxed{\;\langle |\psi_q|^2\rangle=\frac{1}{q\left[\alpha+\beta\left(\frac{q^2+r_eN}{2k}\right)\right]}\;}\tag{★}}$$

## How this can be applied here 

the compression will be based on a wavelet basis, treat each scale j as a band of wave numbers q~2^j.

Let $${cj,k(z)}$$ be wavelet coefficients of $${\psi(\cdot,z)}$$ and define the per scale energy:

$${E_j(z)=\sum_k |c_{j,k}(z)|^2 \;\approx\; \sum_{q\in\text{band}(j)} |\psi_q(z)|^2.}$$

for each scale being: $${\widehat{E}j(z)\;\propto\;\sum{q\in\text{band}(j)} \frac{1}{q\left[\alpha+\beta\left(\frac{q^2+r_eN}{2k}\right)\right]}.\tag{†}}$$

a and b  estimate is not needed - maintain exponential moving averages of E_j and fit a and b by least squares to above

the squared reconstructuon error is upper-bounded by the dropped energy:

$${\|\Delta\psi\|^2 \;\le\; \sum_{j}\!\!\sum_{k:|c_{j,k}|<\varepsilon_j}\! |c_{j,k}|^2 \;\approx\; \sum_{j} \mathbb{E}\!\big[E_j^{\text{drop}}(\varepsilon_j)\big].}$$

for small errors the fidelity satisfied the small angle bound

$${F(\psi,\tilde\psi)=|\langle\psi|\tilde\psi\rangle|^2 \;\gtrsim\; 1-\tfrac{1}{2}\|\Delta\psi\|^2.
\tag}$$ {⋆}``

to target F per gate 
$${\sum_j \mathbb{E}\!\big[E_j^{\text{drop}}(\varepsilon_j)\big]\;\le\; 2(1-F_\star).\tag}$$ {C}

Compute $${\mathbb{E}[E_j^{\text{drop}}]}$$ if coefficients at scale j are modeled as complex zero-mean with variance $${\sigma_j^2\!\approx\!\widehat{E}j/(\#k)}$$ then $${|c{j,k}|^2}$$ is exponential with mean $${\sigma_j^2}$$ (Rayleigh amplitude).

the expected dropped energy with hard-thresholding at $${\varepsilon_j}$$

$${\mathbb{E}\!\big[E_j^{\text{drop}}\big] = (\#k)\,\mathbb{E}\big[|c|^2 \mathbf{1}\{|c|<\varepsilon_j\}\big] = (\#k)\,\sigma_j^2\left(1-e^{-\varepsilon_j^2/\sigma_j^2}\Big(1+\frac{\varepsilon_j^2}{\sigma_j^2}\Big)\right). \tag{D}}$$

Solve C for $${\varepsilion_j}$$

### Uniform fidelity budget per scale

allocate equally across J active scales:

$${\mathbb{E}[E_j^{\text{drop}}]\le \frac{2(1-F_\star)}{J}\quad\Rightarrow\quad \varepsilon_j=\sigma_j\cdot u^{-1}\!\Big(\frac{2(1-F_\star)}{J\,(\#k)\,\sigma_j^2}\Big),}$$

where $${u(x)=1-e^{-t}(1+t)}$$ with $${t=\varepsilon_j^2/\sigma_j^2}$$. (invert numerically; it's a 1D monotone function)

### Information weighted budget 

using the canonical prior to protect banks with larger $${\partial T/\partial E_j}$$.

Allocate:

$${\mathbb{E}[E_j^{\text{drop}}]\propto \frac{1}{w_j},\quad w_j \approx \Big(\frac{q_j^2+r_eN}{2k}\Big) \quad(\text{from }T\text{ weight}),}$$

so high-q banks (which feed T) get stricter thresholds.

online controller after each gate:

$${\boxed{\;\varepsilon_j^{(t+1)} \;=\; \mathrm{clip}\!\left(\varepsilon_j^{(t)}\cdot \exp\big(\gamma\,[F^{(t)}-F_\star]\big),\;\varepsilon_{\min},\varepsilon_{\max}\right)\;}}$$

if measured $${F^{(t)}<F_\star}$$, the exponent is negative -> thresholds tighten if $${F^{{t}}}$$ exceeds target by a margin they loosen. Choose $${\gamma\in[0.1,1]}$$ and set $${\varepsilon_{\min}}$$ from numerical precision and $${\varepsilon_{\max}}$$ from max drop tolerance.