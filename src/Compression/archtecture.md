# Compression Step

    Remove redundent edges
            |
            V
    Clifford Equivalence
            |
            V
    Tensor contraction simplications

Next is a step that is not shown in the diagram, but is an important part of the process:
Clifford Algebra simplifications and tensor contraction simplifications.

Next Compressed Graph Latent (Q-G')

	|G\rangle: Full graph state
	|G’\rangle: Compressed graph state such that |G\rangle \approx U |G’\rangle

    G \xrightarrow[]{\text{compression}} G’ \quad \text{such that} \quad f_{\text{task}}(G) \approx f_{\text{task}}(G’)

And then:

    G’ \xrightarrow[]{\text{diffusion+decoder}} \hat{x}

