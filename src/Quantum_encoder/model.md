# 🧠 Hybrid Quantum-Classical Model with Parametrized Quantum Circuits

This project implements a hybrid quantum-classical machine learning model using **variational quantum circuits (VQCs)**. It combines quantum state preparation, a unitary ansatz (model), and classical gradient-based optimization.

---

## 📌 Overview

This model follows three key stages:

1. **Classical-to-Quantum State Preparation**
2. **Model Construction via Parametrized Unitary Circuits**
3. **Training with Classical Optimization of Quantum Expectation Values**

---

## 1. 🧬 State Preparation: Encoding Classical Data into Quantum States

To utilize quantum computation for machine learning, we begin by **embedding classical input** $\mathbf{x} \in \mathbb{R}^n$ into a quantum state $|\phi(x)\rangle$.

Several encoding methods exist, such as:

* **Angle encoding**: $x_i \rightarrow R_y(x_i)$
* **Amplitude encoding**: $\mathbf{x} \rightarrow \sum_i x_i |i\rangle$

here we are using a hybrid state to convert 

Resulting quantum state:

$$
|\phi(x)\rangle
$$

---

## 2. 🔁 Model Circuit: The Parametrized Quantum Ansatz

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

## 3. 📏 Measurement and Prediction

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

## 📦 Summary

| Step                | Description                             |                  |
| ------------------- | --------------------------------------- | ---------------- |
| Classical → Quantum | Encode input as (                       | \phi(x)\rangle ) |
| Model               | Apply unitary ansatz $U(\theta)$        |                  |
| Measurement         | Compute ( P(q\_0 = 1                    | x; \theta) )     |
| Gradient            | Use observable-based gradient rule      |                  |
| Training            | Optimize parameters using classical SGD |                  |

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
