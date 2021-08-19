export BIFMHelper, functional_dependencies

@doc raw"""
The BIFMHelper node is a node required to perform efficient message passing inconjuction with
the BIFM node. It is required to switch from the backward pass with messages to the forward pass
with marginals.

```julia
out ~ BIFMHelper(in)
```

Interfaces:
1. out - output of the BIFMHelper node, should be connected to the state space model.
2. in - input of the BIFMHelper node, should be connected to the prior for the latent state.
"""
struct BIFMHelper <: AbstractFactorNode end

@node BIFMHelper Stochastic [ out, in ]

# specify custom functional dependencies for BIFMHelper node
function functional_dependencies(dependencies, factornode::FactorNode{ <: Type{BIFMHelper} }, iindex::Int)
    cindex             = clusterindex(factornode, iindex)

    nodeinterfaces     = interfaces(factornode)
    nodeclusters       = factorisation(factornode)
    nodelocalmarginals = localmarginals(factornode)

    varcluster = @inbounds nodeclusters[ cindex ]
    
    # output
    if iindex === 2
        mdependencies = (nodeinterfaces[1], )
        return tuple(mdependencies...), ()
    elseif iindex === 1
        qdependencies = TupleTools.deleteat(nodelocalmarginals, cindex)
        return (), tuple(qdependencies...) 
    end
    
end

@average_energy BIFMHelper (q_out::Any, q_in::Any) = begin
    return entropy(q_in)
end