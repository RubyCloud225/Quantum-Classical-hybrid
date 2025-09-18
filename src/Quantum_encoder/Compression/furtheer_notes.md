# Further Work a Space - Wave compression

## 1 — Schrödinger equation (clean)

$${i\hbar\frac{\partial}{\partial t}\,|\psi(t)\rangle = H\,|\psi(t)\rangle.}$$

From this you can define the time-derivative vector or the Hamiltonian action interchangeably:

$${|w\psi(t)\rangle \;:=\; i\hbar\frac{\partial}{\partial t}|\psi(t)\rangle \;=\; H\,|\psi(t)\rangle .}$$

So $${|w\psi\rangle}$$ is exactly the Hamiltonian applied to the state (up to the Schrödinger identity).

⸻

## 2 — Define the W-weighted momentum scalar

$${W_{p,\text{total}}(t) \;:=\; \langle w\psi(t)\,|\,\hat p\,|\,w\psi(t)\rangle
\;=\; \langle \psi(t)\,|\,H\,\hat p\,H\,|\,\psi(t)\rangle }$$

since $${|w\psi\rangle = H|\psi\rangle.}$$

(This is a scalar — the momentum weighted by H-action.)

⸻

## 3 — Decompose into kept / vacuum / cross with projector P

Let P be the projector onto the (compressed) subspace and Q=I-P. Insert I=P+Q around $${\hat p}$$:

$${\begin{aligned}
W_{p,\text{total}}
&= \langle w\psi|(P+Q)\,\hat p\,(P+Q)|w\psi\rangle \\
&= \underbrace{\langle w\psi|P\hat pP|w\psi\rangle}{W{p,\text{kept}}}
	•	\underbrace{\langle w\psi|Q\hat pQ|w\psi\rangle}{W{p,\text{vac}}} \\
&\qquad + \underbrace{\langle w\psi|P\hat pQ|w\psi\rangle + \langle w\psi|Q\hat pP|w\psi\rangle}{W{p,\text{cross}}}.
\end{aligned}}$$

Write the cross term compactly:
$${W_{p,\text{cross}} = 2\Re\big(\langle w\psi|P\hat pQ|w\psi\rangle\big).}$$

So the final identity is
$${\boxed{\,W_{p,\text{total}} = W_{p,\text{kept}} + W_{p,\text{vac}} + W_{p,\text{cross}}\,.}}$$

If you want a conditional (normalized) W-weighted momentum inside the kept subspace, divide by the kept weight for |w\psi\rangle:
$${w_{\text{kept}} := \langle w\psi|P|w\psi\rangle,\qquad
W_{p,\text{eff}} = \frac{W_{p,\text{kept}}}{w_{\text{kept}}}
\quad(\text{if }w_{\text{kept}}\neq0).}$$

Note: because $${|w\psi\rangle=H|\psi\rangle,}$$ you can evaluate these by first forming the vector $${ H|\psi\rangle}$$ (or numerically approximating $${i\hbar\partial_t|\psi\rangle}$$ ), 

⸻

## 4 — Relation to time evolution inside the compressed subspace

If you define an effective Hamiltonian in the compressed subspace
$${H_{\text{eff}} := P H P}$$,
then a common approximation for dynamics restricted to the subspace is
$${|\psi_c(t)\rangle \approx e^{-\,\tfrac{i}{\hbar}H_{\text{eff}}\,t}\;|\psi_c(0)\rangle,
\qquad\text{where }|\psi_c(0)\rangle=\frac{P|\psi(0)\rangle}{\|P|\psi(0)\rangle\|}}$$.

Two important caveats:
	•	This neglects coupling (coherences) between kept and discarded subspaces. Those couplings are exactly encoded in the cross term $${W_{p,\text{cross}}}$$ (and more generally by PHQ terms). If cross terms are non-negligible you must include correction terms (e.g., Feshbach projection formalism) or enlarge the kept subspace.
	•	The W-weighted momentum you defined evolves according to the Heisenberg or Schrödinger equations; if you want $${W_{p,\text{total}(t)}}$$ at later time, compute $${H|\psi(t)\rangle}$$ and evaluate the same decomposition at that t.

So a practical formula for your “final” (compressed) state evolution is:

$${|\psi_{\text{final}}(t)\rangle \approx e^{-\,\tfrac{i}{\hbar}H_{\text{eff}}\,t}\;P\,|\psi(0)\rangle}$$
and you should treat corrections proportional to PHQ (hence $${W_{p,\text{cross}}}$$) as a diagnostic of approximation error.

⸻

## 5 — Putting it into computation (how to evaluate numerically, GPU/C++ friendly)
	1.	Compute $${|\psi(t)\rangle}$$ by applying your unitary (C++/CUDA).
	2.	Compute $${|w\psi\rangle = H\,|\psi(t)\rangle}$$.
	•	If H is available as an operator-action: apply it (e.g. split-step / kinetic-multiplication in momentum basis + potential in position basis).
	•	Alternatively approximate $${i\hbar \partial_t|\psi\rangle}$$ numerically if you have time-series data.
	3.	To evaluate W-weighted momentum terms do these in momentum basis:
	•	FFT $${|w\psi\rangle \to \widetilde{w\psi}(p)}$$ (cuFFT).
	•	Elementwise multiply by p and integrate:
$${W_{p,\text{kept}} = \sum_{p} p\,\big|\widetilde{w\psi}_P(p)\big|^2}$$
where $${\widetilde{w\psi}_P}$$ is the FFT of the kept-projected $${P|w\psi\rangle}$$. Do the same for Q and cross (cross computed as $${2\Re\sum p\,\overline{\widetilde{w\psi}_P}\,\widetilde{w\psi}_Q}$$).
	4.	Compute diagnostics: vacuum weight $${w_{\text{vac}}=\langle w\psi|Q|w\psi\rangle, cross ratio $$|W_{p,\text{cross}}|/|W_{p,\text{total}}|$$, etc.
	5.	If cross ratio is small, the $${H_{\text{eff}}$$-driven evolution is a good approximation; otherwise iterate (increase retained modes or change optimization of U).

⸻

6 — Short cleaned summary you can paste into docs
	•	Define $${|w\psi\rangle := H|\psi\rangle = i\hbar\partial_t|\psi\rangle}$$.
	•	W-weighted total momentum:
$${W_{p,\text{total}} = \langle w\psi|\hat p|w\psi\rangle}$$.
	•	Decompose with P+Q=I:
$${W_{p,\text{total}} = W_{p,\text{kept}} + W_{p,\text{vac}} + 2\Re\langle w\psi|P\hat pQ|w\psi\rangle}$$.
	•	Conditional kept value:
$${W_{p,\text{eff}} = \frac{W_{p,\text{kept}}}{\langle w\psi|P|w\psi\rangle}}$$.
	•	Time-evolve compressed state with $${H_{\text{eff}}=PHP}$$ and treat cross terms as the error term.

## Projection (Feshbach) correction — derivation and formulas

Partition Hilbert space with orthogonal projectors P (kept) and Q=I-P (discarded). Start from the stationary Schrödinger equation
$${H|\Psi\rangle = E|\Psi\rangle,
\qquad |\Psi\rangle = P|\Psi\rangle + Q|\Psi\rangle}$$.

Projecting yields two coupled equations
$${\begin{aligned}
P H P \, P|\Psi\rangle + P H Q \, Q|\Psi\rangle &= E\, P|\Psi\rangle,\\
Q H P \, P|\Psi\rangle + Q H Q \, Q|\Psi\rangle &= E\, Q|\Psi\rangle.
\end{aligned}}$$

Solve the second for $${Q|\Psi\rangle}$$ (formal resolvent):
$${Q|\Psi\rangle \;=\; \big(E - Q H Q\big)^{-1}\,Q H P \,P|\Psi\rangle}$$,
(substitute back into the first) to get the energy-dependent effective Hamiltonian acting on the P-subspace:
$${\boxed{\,H_{\text{eff}}(E)
= P H P \;+\; P H Q \,\big(E - Q H Q\big)^{-1}\,Q H P\,.}
\tag{Feshbach}}$$

The second term $${\Sigma(E)=P H Q (E-QHQ)^{-1}}$$ Q H P is the self-energy / correction from virtual excursions into the discarded subspace. If $${\Sigma}$$ is small the naive P H P was fine; otherwise \Sigma can be large and must be included.

## 7 — Algorithm with Projection Correction

To implement the projection correction in practice, follow these steps:

1. **Compute the state vector $|\psi(t)\rangle$** by applying the time evolution unitary operator using your C++/CUDA implementation.

2. **Compute the weighted vector $|w\psi\rangle = H|\psi(t)\rangle$**, either by applying the Hamiltonian operator directly or by numerical differentiation if time-series data is available.

3. **Project $|w\psi\rangle$ into kept and vacuum subspaces** using projectors \(P\) and \(Q = I - P\):
   - Compute \(P|w\psi\rangle\) and \(Q|w\psi\rangle\).

4. **Compute the W-weighted momentum components**:
   - \(W_{p,\text{kept}} = \langle w\psi|P \hat p P|w\psi\rangle\),
   - \(W_{p,\text{vac}} = \langle w\psi|Q \hat p Q|w\psi\rangle\),
   - \(W_{p,\text{cross}} = 2\Re\langle w\psi|P \hat p Q|w\psi\rangle\),
   using FFTs and elementwise multiplications as described previously.

5. **Construct the effective Hamiltonian with projection correction**:
   \[
   H_{\text{eff}}(E) = P H P + \Sigma(E),
   \]
   where the self-energy correction \(\Sigma(E)\) is given by
   \[
   \Sigma(E) = P H Q (E - Q H Q)^{-1} Q H P.
   \]

6. **Compute \(\Sigma(E)\) by solving linear systems**:
   - For each basis vector \(u_j\) in the kept subspace, compute
     \[
     x_j = (E - Q H Q)^{-1} Q H P u_j,
     \]
     by solving the linear system
     \[
     (E - Q H Q) x_j = Q H P u_j.
     \]
   - Then assemble \(\Sigma(E)\) as
     \[
     \Sigma(E) = \sum_j P H Q x_j \langle u_j| \cdot \rangle.
     \]

7. **Use \(H_{\text{eff}}(E)\) to evolve the compressed state**:
   \[
   |\psi_c(t)\rangle \approx e^{-\,\tfrac{i}{\hbar}H_{\text{eff}}(E)\,t} |\psi_c(0)\rangle,
   \]
   where \(\Sigma(E)\) accounts for corrections due to interactions with the vacuum and cross terms, improving the accuracy beyond the naive \(P H P\) approximation.

---

### Summary Algorithm

- Compute \( |\psi(t)\rangle \) via C++/CUDA unitary application.
- Compute \( |w\psi\rangle = H|\psi(t)\rangle \).
- Project \( |w\psi\rangle \) into kept (\(P\)) and vacuum (\(Q\)) subspaces.
- Evaluate \( W_{p,\text{kept}}, W_{p,\text{vac}}, W_{p,\text{cross}} \) for diagnostics.
- Construct corrected effective Hamiltonian \( H_{\text{eff}}(E) = P H P + \Sigma(E) \).
- Compute \(\Sigma(E)\) by solving linear systems \((E - Q H Q)x = Q H P u_j\).
- Evolve compressed state with \(H_{\text{eff}}(E)\) incorporating projection corrections.
