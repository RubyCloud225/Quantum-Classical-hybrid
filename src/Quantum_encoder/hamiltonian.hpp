// first step Provide a base class for Hamiltonian

// Calculate Transmon k 
// matrix the single transmon hamiltonian k = 4E_cn^2 - E_j\cos\varphi
// keep the lowest eigenlevels of the matrix
// approximate the hamiltonian with a truncated basis H_k \approx \hbar\omega_k a_k^\dagger a_k + \tfrac{\alpha_k}{2} a_k^\dagger a_k^\dagger a_k a_k.
//  H_0 = \sum_{k=1}^{N} \Big(\sum_{m=0}^{d-1} E_m^{(k)} |m\rangle\langle m| \Big) + \sum_{\langle k,\ell\rangle} H_{\text{int}}^{k\ell}.
// First step \Big(|sum_{m=0}^{d-1} E_m^{(k)} |m\rangle\langle m| \Big)

// Second step |sum_{\langle k,\ell\rangle} H_{\text{int}}^{k\ell}.
