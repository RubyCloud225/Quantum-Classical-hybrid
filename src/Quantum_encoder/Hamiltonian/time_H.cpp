// Compute U_k = exp(-i H dt) for Hermitian H using Jacobi diagonalization
//
// Equations:
// H^(k) = H0 + sum_i [ u_X,i^(k) H_X,i + u_Z,i^(k) H_Z,i ]
// Jacobi diagonalization: H = V Λ V†
// U = V diag( e^{-i λ_m dt} ) V†

// Hermitian Symmetrization: A + A† / 2 - bring in from H_map.cpp
// Jacobi rotation for diagonalization for the hermitian matrix
// Jacobi eigen - decomposition: H -> V Λ V†
// Multiply by exp(-i λ_m dt) for each eigenvalue λ_m
// Conjugate transpose V†
// Multiply U_k = V diag( e^{-i λ_m dt} ) V†
// Apply U to Psi