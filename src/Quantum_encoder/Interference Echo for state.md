# QFT Interference Echo for State Preparation

This document describes a method to measure and include the interference "echo" of Gaussian wavepackets in 1+1D scalar Quantum Field Theory (QFT), suitable for integration at the state preparation stage of a qubit circuit.

## 1. 1+1D Free Scalar Field

The scalar field operator is:

```
phi(x,t) = Integral[ dk / sqrt(4 * pi * omega_k) * (a_k * exp(-i*(omega_k*t - k*x)) + a_k^dagger * exp(i*(omega_k*t - k*x))) ]
```

where:

* omega\_k = sqrt(k^2 + m^2)
* \[a\_k, a\_k^dagger] = delta(k-k')

---

## 2. Gaussian Wavepackets

Define wavepackets for state preparation:

```
|phi_j> = Integral[ dk * f_j(k) * a_k^dagger |0> ],  j = 1,2
```

with Gaussian momentum distributions:

```
f_j(k) = (1 / (2 * pi * sigma^2)^(1/4)) * exp[-(k - k_j)^2 / (4*sigma^2) + i*k*x_j]
```

Parameters:

* k\_j: central momentum
* x\_j: initial position
* sigma: momentum width

---

## 3. Interference Echo Measurement

The interference (echo) is captured by the cross-term:

```
E(x,t) = <phi_1 | phi(x,t) | phi_2>
       = Integral[ dk / sqrt(4*pi*omega_k) * f_1^*(k) * f_2(k) * exp(-i*(omega_k*t - k*x)) ]
```

* Represents the residual correlation between wavepackets.
* Encodes interference and propagation effects.

---

## 4. Simplification (Narrow Gaussian & Massless Limit)

For sigma << k\_1, k\_2 and m -> 0:

```
omega_k ~ |k|
E(x,t) ~ Sum_{k near k_j} f_1^*(k) * f_2(k) * exp(-i*(|k|*t - k*x))
```

* The echo moves roughly at speed of light: x \~ ± t.
* Retains spatial-temporal interference structure.

---

## 5. Total Echo Strength (Sanity Measurement)

A single metric to monitor echo strength:

```
E_tot(t) = Integral[ dx * |E(x,t)|^2 ]
```

* E\_tot(t) \~ 1: wavepackets overlap constructively (coherent state)
* E\_tot(t) \~ 0: wavepackets have decohered (minimal interference)

This can serve as a **sanity check** at the state preparation stage of a qubit circuit.

---

## 6. Integration Note

* Compute `E(x,t)` or `E_tot(t)` immediately after preparing the wavepackets.
* Use `E_tot(t)` to tune amplitudes, phases, and positions in your qubit state preparation routine.
* This ensures that the QFT-based interference pattern is faithfully represented in the quantum circuit.

---

**Reference:**

* Standard 1+1D scalar field theory, Gaussian wavepacket propagation, and two-point correlation functions.
