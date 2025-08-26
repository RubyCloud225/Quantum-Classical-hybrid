# Hybrid Quantum-Classical Model with Parametrized Quantum Circuits

This project implements a hybrid quantum-classical machine learning model using **variational quantum circuits (VQCs)**. It combines quantum state preparation, a unitary ansatz (model), and classical gradient-based optimization.

---

## Overview

This model follows three key stages:

1. **Classical-to-Quantum State Preparation**
2. **Model Construction via Parametrized Unitary Circuits**
3. **Training with Classical Optimization of Quantum Expectation Values**

---

## 1. State Preparation: Encoding Classical Data into Quantum States

To utilize quantum computation for machine learning, we begin by **embedding classical input** $\mathbf{x} \in \mathbb{R}^n$ into a quantum state $|\phi(x)\rangle$.

Several encoding methods exist, such as:

* **Angle encoding**: $${x_i \rightarrow R_y(x_i)}$$
* **Amplitude encoding**: $${\mathbf{x} \rightarrow \sum_i x_i |i\rangle}$$

here we are using a hybrid state to convert 

Resulting quantum state:

$$ {|\phi(x)\rangle}$$

---

## 2. Model Circuit: The Parametrized Quantum Ansatz

We define a **trainable quantum circuit** (also called an *ansatz*) using unitary operators:

$$
U(\boldsymbol{\theta})|\phi(x)\rangle
$$

Where:

* $U(\boldsymbol{\theta})$ is a unitary operator parametrized by vector $\boldsymbol{\theta}$
* $|\phi(x)\rangle$ is the input quantum state
* The output is a quantum state from which measurements can be taken

### Decomposition of U(\u03b8)

We decompose $U(\boldsymbol{\theta})$ into a sequence of **elementary gates**:

$$
U(\boldsymbol{\theta}) = U_L \cdots U_2 U_1
$$

Each $U_i$ may consist of:

* **Single-qubit rotations** $U(\theta, \phi, \lambda)$
* **CNOT gates** for entanglement

### Single-Qubit Rotation Gate

$$
U(\theta, \phi, \lambda) =
\begin{bmatrix}
\cos(\theta/2) & -e^{i\lambda} \sin(\theta/2) \\
\e^{i\phi} \sin(\theta/2) & e^{i(\lambda + \phi)} \cos(\theta/2)
\end{bmatrix}
$$

---

## 3. Measurement and Prediction

After applying the quantum model, measurement is performed to extract probabilities:

$$
f(x; \boldsymbol{\theta}) = P(q_0 = 1 \mid x; \boldsymbol{\theta}) = \sum_{k=1}^{n} \left| (U(\boldsymbol{\theta}) |\phi(x)\rangle)_k \right|^2
$$

We often simplify this as:

$$
P(q_0 = 1 \mid x; \boldsymbol{\theta})
$$

---

## Training via Stochastic Gradient Descent

We optimize the model classically using **stochastic gradient descent (SGD)**.

### Loss Gradient:

$$
\nabla_{\theta} L(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \nabla_{\theta} \pi(x_i; \boldsymbol{\theta}) \cdot \partial_1 l(\pi(x_i; \boldsymbol{\theta}), y_i)
$$

Where:

* $l$ is a classical loss function (e.g., MSE or cross-entropy)
* $\pi(x; \theta) = f(x; \theta)$ is the quantum model’s prediction

---

## Quantum Gradient Derivation

We compute gradients based on **expectation values** of observables like the Pauli-Z operator $\sigma_z$:

$$
\nabla_{\theta} \pi(x; \theta) = -\frac{1}{2} \nabla_{\theta} \langle \phi(x)| U^{\dagger} \sigma_z U | \phi(x) \rangle
$$

Let $\nu$ be a specific parameter in $\boldsymbol{\theta}$. Then:

$$
\partial_{\nu} \pi(x; \theta) = -\text{Re}\left\{ \langle \phi(x) | \partial_{\nu} U^{\dagger} \sigma_z U | \phi(x) \rangle \right\}
$$

---

## Unitary Gate Parameter Gradients

Each elementary gate contributes to the derivative of $U(\boldsymbol{\theta})$:

$$
\partial_{\nu} U = U_1 \cdots \partial_{\nu} U_i \cdots U_L
$$

### Specific Gate Rules:

1. **Rotation \u03b8:**

   $$
   \partial_{\theta} U = \frac{1}{2} U(\theta + \pi, \phi, \lambda)
   $$

2. **Rotation \u03c6 or \u03bb:**

   $$
   \partial_{\phi} U = \frac{i}{2} \left( U(\theta, \phi, \lambda) - U(\theta, \phi + \pi, \lambda) \right)
   $$

---

## 🔍 Final Model Gradient Expression

Gradient becomes:

$$
\sum_{k=1}^{K} a_k \cdot \text{Re} \left\{ \langle \phi(x) | U(\theta_k)^{\dagger} \sigma_z U(\theta) | \phi(x) \rangle \right\} \\
+ \sum_{l=1}^{L} b_l \cdot \text{Im} \left\{ \langle \phi(x) | U(\theta_l)^{\dagger} \sigma_z U(\theta) | \phi(x) \rangle \right\}
$$

---

## Hamiltonian Mapping Layer

The physical device is modeled as:

$$
\H(t) = H_0 + \sum_{j} u_j(t) H_{c,j}
$$

where $H_0$ is the drift hamiltonian (static ), and $H_{c,j}$ are the control hamiltonians (dynamic).
The control amplitudes $u_j(t)$ are the control parameters.

time evolution:
$$
\U_T = \mathcal{T} \exp\left( -\frac{i}{\hbar} \int_0^T H(t) \, dt \right)
$$

where $\mathcal{T}$ is the time-ordering operator.
The time-evolution operator $\U_T\$ is the solution to the time-dependent Schröder equation.

where:

## Single Transmon (Duffing model)

For transmon i:

$${H_{\text{transmon},i} = \omega_i\, a_i^\dagger a_i •	\frac{\alpha_i}{2} \, a_i^\dagger a_i^\dagger a_i a_i}$$

where:

	•	${\omega_i}$ = fundamental transition frequency of the qubit (GHz in simulation)
	•	${\alpha_i < 0}$ = anharmonicity (negative for transmons)
	•	${a_i, a_i^\dagger}$ = annihilation / creation operators for mode i

In number-operator form (numerically stable):

${a_i^\dagger a_i^\dagger a_i a_i = n_i (n_i - 1) with n_i = a_i^\dagger a_i.}$

⸻

## Coupling Between Qubits

For two transmons i and j, a simple exchange (hopping) coupling:

$${H_{\text{coupling},ij} = g_{ij} \left(a_i^\dagger a_j + a_j^\dagger a_i\right)}$$ 

${g_{ij}}$ = coupling strength (GHz)
This produces an XY-type interaction in the computational subspace.

⸻

## Drift Hamiltonian ${H_0}$

The system drift is the sum of single-transmon Hamiltonians plus all couplings:

$${H_0 = \sum_{i=1}^N \left[ \omega_i\, n_i + \frac{\alpha_i}{2} n_i(n_i - 1) \right] •	\sum_{i<j} g_{ij} \left(a_i^\dagger a_j + a_j^\dagger a_i\right)}$$

This ${H_0}$ is time-independent and captures the fixed device physics.

⸻

## Control Hamiltonians \{H_{c,j}\}

We define a set of time-dependent control channels that the RL agent will modulate.
	•	X-drive on qubit i:
$${H_{X,i} = a_i + a_i^\dagger}$$

corresponding (in RWA) to a resonant microwave drive that generates X-axis rotations.
	•	Z-control on qubit i:

$${H_{Z,i} = n_i}$$
corresponding to flux detuning / Stark shifts that modulate the Z-axis rotation rate.

The full time-dependent Hamiltonian is:

$${H(t) = H_0 + \sum_{i=1}^N \left[ u_{X,i}(t) H_{X,i} + u_{Z,i}(t) H_{Z,i} \right]}$$

where ${u_{X,i}(t), u_{Z,i}(t)}$ are control amplitudes (to be learned).

extend this to a physical lab RWA rotating frame 

generate a carrier frequency for each qubit at the desired frequency

this equates to $${H_{\text{lab}, i}(t) = \varepsilion_i(t)cos(\omega_{d,i} t + \phi_i(t)) (a_i + a_i^\dagger)}$$

where $\omega_{d,i}$ is desired frequency, $\varepsilion_i(t)$ is the amplitude, and $\phi_i(t)$ is the phase of the drive.

$${\mathcal{E}_i(t) = \tfrac{1}{2}\varepsilon_i(t) e^{-i\phi_i}}$$

then RWA = $${H_{\text{lab}, i}(t) = \mathcal{E}_i(t)(a_i^\dagger) = \mathcal{E}_i^*(t)(a_i)}$$

in our case we are looking at working with qubits in a rotating frame, so we can set 

$${u_{X,i}(t) \equiv 2\Re[\mathcal{E}i(t)] \quad\text{and}\quad}$$

and

$${u{Y,i}(t) \equiv -2\Im[\mathcal{E}_i(t)]}$$

so we can represent the in phase / quadrature pair with the same a + a^dagger and i(a - a^dagger) operators

## Time-dependent Simulation 

following through we then divide total control time $\text{T}$ into $\text{M}$ time steps of duration $\Delta t = \text{T}/\text{M}$

on slice $\text{k}$ we have assumed the controls constant: 
$$\mathcal{u}_{x,u}(t) = \mathcal{u}_{x,u}^{(k)}$$ for $$t\in[t_k,t_k+\Delta t)$$

then: 
$${H^{(k)} = H_0 + \sum_{i} \big( u_{X,i}^{(k)} H_{X,i} + u_{Z,i}^{(k)} H_{Z,i} \big)}$$

Propagator $\text{u}(k)$ for Hermitian ${H^(k)}$:

use a spectral decomposition:

- Diagonalize ${H^{(k)}}$:

$${H^{(k)} = V^{(k)} \Lambda^{(k)} \left( V^{(k)} \right)^\dagger}$$

where ${V^{(k)}}$ contains eigenvectors and ${\Lambda^{(k)}}$ is diagonal with eigenvalues^${\lambda_m^{(k)}}$.

- Exponentiate the Diagonal:
$${D^{(k)} = \mathrm{diag}\left( e^{-i \lambda_1^{(k)} \Delta t}, \dots, e^{-i \lambda_D^{(k)} \Delta t} \right)}$$

- Recompose
$${U^{(k)} = V^{(k)} D^{(k)} \left ( V^{(k)} \right)^\dagger}$$

State update:

$${|\psi_{k+1}\rangle = U_k\,|\psi_k\rangle,\qquad U_{\text{tot}} = U_M \cdots U_2 U_1}$$

Target: given $${\psi_{\text{target}}}$$ from your VQE / gate layer, maximize fidelity

$${\mathcal{F} = |\langle\psi_{\text{target}} | \psi_T\rangle|^2}$$


⸻

## Fidelity as Reward

The training reward is based on state fidelity:

$${\mathcal{F} = \left|\langle \psi_{\text{target}} | \psi_T \rangle\right|^2}$$

plus optional penalties for leakage, energy, or bandwidth.

---

## RL Model

state RL agent: 
- Slice index k 
- selected observables $\langle Z\rangle, \langle X\rangle$ overlap with $\psi_{\text{target}}$
- selected control parameters $u_j$

action RL agent:
- select next control parameters $u_j$

reward 

Terminal: fidelity $\F = |\langle\psi_{\text{target}} | \psi_T\rangle|^2$

- Optional shaping: leakage penalty, energy cost

Episode:

Reset to $\|0\dots0\rangle$

For each time slice:
RL outputs $\u_j$

Simulate $\psi \to e^{-iH(t_k)\Delta t} \psi$

Return reward to update policy

---

## Gate -> Hamiltonian Mapping

- Gate set: $\{U_1, U_2, \dots, U_n\}$
- Hamiltonian set: $\{H_1, H_2, \dots, H_n \}$

To connect your circuit to hardware controls:

${R_x(\theta) = e^{-i(\theta/2)\sigma_x} → H_x = \sigma_x/2}$
${R_z(\phi) = e^{-i(\phi/2)\sigma_z} → H_z = \sigma_z/2}$

CNOT: generated via interaction term ${H_{\text{CX}}}$ (e.g., cross-resonance Hamiltonian)

These mappings allow an initial guess for ${u_j(t)}$ or curriculum learning.

⸻


## 📦 Summary

| Step                | Description                             |                  |
| ------------------- | --------------------------------------- | ---------------- |
| Classical → Quantum | Encode input as (                       | \phi(x)\rangle ) |
| Model               | Apply unitary ansatz $U(\theta)$        |                  |
| Measurement         | Compute ( P(q\_0 = 1                    | x; \theta) )     |
| Gradient            | Use observable-based gradient rule      |                  |
| Training            | Optimize parameters using classical SGD |                  |
| Gate -> Hamiltonian | Map gate set to Hamiltonian set         |                  |

---

## 🧠 Applications

* Quantum classifiers
* Quantum kernel machines
* Variational Quantum Eigensolvers (VQE)
* Quantum Generative Models

---

## 📚 References

* Schuld, M., Bocharov, A., Svore, K.M., & Wiebe, N. (2018). *Circuit-centric quantum classifiers*.
* Mitarai, K., Negoro, M., Kitagawa, M., & Fujii, K. (2018). *Quantum circuit learning*.
* Nielsen, M. A., & Chuang, I. L. (2000). *Quantum Computation and Quantum Information*.
* Liu, Y (2025). *Superconducting quantum computing optimisation based on multi-bjective deep reinforcmenet learning*.
