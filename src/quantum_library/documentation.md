https://pmc.ncbi.nlm.nih.gov/articles/PMC9955545/pdf/entropy-25-00287.pdf 

Principles to focus on is superposition, entanglement and quantum measurement  . 

The superposition principle states that a quantum system can exist in multiple states 
simultaneously. This is in contrast to classical systems, where a system can only be in one
state at a time. The superposition principle is a fundamental concept in quantum mechanics.

Entanglement is a phenomenon where two or more particles become correlated in such a way
that the state of one particle cannot be described independently of the others, even when
the particles are separated by large distances.

Quantum measurement is the process of determining the state of a quantum system, which is a
fundamental aspect of quantum mechanics.

use vectors and matrices 

transforming classicial data to quantum data 

Quantum data is represented as a vector in a complex vector space, while classical data is
represented as a vector in a real vector space. The transformation from classical to quantum
data is a fundamental concept in quantum computing.

HHL Algorithm: Mx = v

looking at the matrix inversion problem with matrix and vector to determin the vector x

to determin the parameter we use the matrix inversion algorithm
 M = input characteristics of data points in matrix X 
 v = input vector of data points in vector y and matrix X

INstead of using linear regression we use the matrix inversion algorithm to determin the
vector x

instead of y = 0^Tx, we will use (X^TX)0 = X^TY
thus M = X^TX and v = X^TY

first identify one or more operators capable of transforming the state |v> into the solution
vector 0.

    M~ =    [0 M^T]
            [M 0]

it has the following eigenvalue decomposition: 
    M~ = Σ λi | ui> <ui|,

 which the eigenvectors |ui> provide a orthonormal basis for the vector space.
 the vector may be expressed as a linear combination of the eigenvectors:
 |v> = ∑ betai |ui>

 this answers the inversion problem Mx = v, where |x> = M~-1 |v>


 to transfer classical data to quantum data we need to perform HHl algorithm
 1.  prepare the quantum register in the state |v>
    apply sequence of gates to qubits to achieve superposition
    use efficient algorithms and to off load the complexity use variational methods
    use quantum parallelism to speed up the computation
 2.  apply the quantum circuit that implements the matrix inversion algorithm
 3.  measure the quantum register in the computational basis to obtain the solution vector
 x>


 following that we follow the linear regression algorithm to determin the parameter of the model

turn binary string to a state n bits

then construct pauli x gate to flip the bits from 1 and 0 to |0> and |1>


encoder is amplitude encoding - data points are encoding into amplitudes of quantum states. 

defines a unitary matrix gate 

|X> = x_n / √∑^n i=1 x^i |n - 1>



