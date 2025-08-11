# Quantum Graph Encoder (Simulated)

    Parameterized unitaries (U(theta, lambda))

                    Compression
                    |
                    V
        Hadamard gate (H)
        |           |
                    V
                    Compression
        |           |
                    v
        CNOT -> 
            |
            v
            Hamiltonian Base - 
        graph Entanglement -> Measurement


to calculate realistic noise models and to simulate the quantum circuit we will circuit largrangian formulation of the quantum circuit. 
The circuit Lagrangian formulation is a mathematical framework that allows us to describe the quantum circuit in a way that is amenable to optimization. 
The circuit Lagrangian formulation is based on the idea of representing the quantum circuit as a sequence of unitary operations, and th ence as a sequence of Lagrangian multipliers.

go from unitary to control pulses by using a transmon device- this is a quantum device that is used to implement quantum gates.

starting with a Hamiltonian - (transmons + couplers) - we can use the control pulses to implement the quantum gates.

this control operators that used to bring together the drives and a optimal - control solver which finds time - dependent coefficients 

    U_T = \mathcal{T}\exp\!\Big(-\tfrac{i}{\hbar}\int_0^T (H_0 + \sum_j c_j(t)H_{c,j})\,dt\Big)

this approximates the unitary operator U_T that implements the quantum gate.

# Step 1 Base Hamiltonian 

    H_0 = \sum_{k=1}^{N} \Big(\sum_{m=0}^{d-1} E_m^{(k)} |m\rangle\langle m| \Big) + \sum_{\langle k,\ell\rangle} H_{\text{int}}^{k\ell}.

for each transmon k_i either diagonalise the single transmon hamiltonian 4E_C n^2 - E_J\cos\varphi and keep the lowest d eigenvalues and eigenvectors

    H_k \approx \hbar\omega_k a_k^\dagger a_k + \tfrac{\alpha_k}{2} a_k^\dagger a_k^\dagger a_k a_k.

Coupling hamiltonian between two transmons k and l 

    H_{\text{int}} = g_{k\ell}(a_k^\dagger a_\ell + a_k a_\ell^\dagger). 

our control operators

drive qubit(k) couples as H_{c,k}(t) = \epsilon_k(t) (a_k + a_k^\dagger)  

for flux tunable couplers - H_coupler(t)

H_{c,k\ell}(t) = \epsilon_{k\ell}(t)

# Step 2 Objective & Constraints

Define fidelity (projected to computational subspace) — let P be projector onto 2^N-dim computational subspace and U_T be full unitary on truncated Hilbert space.
Use projected fidelity

    \mathcal{F} = \frac{1}{d_c}\big|\operatorname{Tr}\!\big( U_{\text{target}}^\dagger\,P\,U_T\,P \big)\big|

    where   d_c=2^N. You can also use state-averaged fidelity.

Add penalties:
	•	Leakage penalty 
        L_{\text{leak}} = \operatorname{Tr}\big(P_\perp U_T \rho U_T^\dagger\big) (population out of computational subspace).
	•	Control regularization: 
        \lambda \sum_j \int_0^T |c_j(t)|^2 dt (energy / amplitude penalty).
	•	Bandwidth constraints by parameterizing c_j(t) in low-bandwidth basis (e.g. Fourier, splines) or add penalty on derivative \int |\dot c_j(t)|^2 dt.

Goal: maximize 

        \mathcal{F} - \beta L_{\text{leak}} - \lambda \|c\|^2.

# step 3 GRAPE Algorithm

Discretize time into M steps, piecewise-constant controls c_{j,k} on [t_k,t_{k+1}). For each step
    
    U_k = \exp\!\big(-i(H_0 + \sum_j c_{j,k} H_{c,j}) \Delta t\big).

Total evolution 
    
    U_T = U_M\dots U_1.

Gradient of fidelity w.r.t a control amplitude at slice k:
    
    \frac{\partial \mathcal{F}}{\partial c_{j,k}} = 2\,\Re\Big\{ \langle \Phi | \, U_M \dots U_{k+1}\, (-i\Delta t H_{c,j})\, U_k \dots U_1 \,|\Psi\rangle \Big\}

(implemented efficiently with forward propagation of states and backward propagation of costates; see standard GRAPE derivation). 

For unitary-target fidelity use adjoint method with
    
    \Lambda_k = U_{k+1}^\dagger \dots U_M^\dagger \,U_{\text{target}} \,.

Then perform gradient ascent with step size or use L-BFGS with gradients and constraints.

Add DRAG correction for single-qubit pulses to reduce |2\rangle leakage:
    \epsilon_{\text{DRAG}}(t)=\epsilon(t) + i\frac{\dot\epsilon(t)}{\Delta}

(where \Delta is the |1\rangle\to|2\rangle detuning).


# Step 4 Implementation

Implement the GRAPE algorithm as a foundation to our simulator

Build finite truncated device Hamiltonian H_0 (each transmon diagonalized to d levels) and control operators H_{c,j}.

Parameterize piecewise-constant controls c_{j,k} on M slices, duration \Delta t.

    For each slice build H_k = H_0 + \sum_j c_{j,k} H_{c,j} and propagator U_k = \exp(-i H_k \Delta t/\hbar).

Compute total U_T = U_M \cdots U_1. Project to computational subspace with projector P and evaluate projected fidelity
    
    \mathcal{F} = \frac{1}{d_c}\big|\operatorname{Tr}(U_{\rm target}^\dagger P U_T P)\big|.

Compute gradients via the GRAPE adjoint method (forward store F_k = U_k\cdots U_1, backward store B_k = U_M\cdots U_{k+1}). For piecewise constant slices, approximate
    
    \frac{\partial U_k}{\partial c_{j,k}} \approx -i\,\Delta t\, H_{c,j}\,U_k,
    and hence
    
    \frac{\partial U_T}{\partial c_{j,k}} = B_k\,(-i\Delta t H_{c,j})\,F_k.
    
So the gradient of 

    \Re\operatorname{Tr}(U_{\rm target}^\dagger P U_T P) w.r.t. c_{j,k} is

    \partial_{j,k}\mathcal{G} = \Re\!\left\{ \operatorname{Tr}\!\Big( U_{\rm target}^\dagger P\, B_k(-i\Delta t H_{c,j})F_k\,P \Big)\right\},
    and scale by 1/d_c if using \mathcal{F}.
	
Update pulses with gradient ascent (or L-BFGS). Project / clip to amplitude limits and smooth or parametrize to enforce bandwidth.