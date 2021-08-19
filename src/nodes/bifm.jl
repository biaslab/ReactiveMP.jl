export BIFM, BIFMMeta, functional_dependencies

@doc raw"""
The BIFM node is a node that can be used as a substitution for a state space model. It includes all factor of the time slice to perform efficient inference.
This node needs to be used in conjuction with the BIFMHelper node for efficient inference.

```julia
out ~ BIFM(in, zprev, znext)
```

Interfaces:
1. out - latent output (observation) of the BIFM node
2. in - latent input of the BIFM node
3. zprev - previous latent state of the BIFM node
4. znext - next latent state of the BIFM node

## Example

```julia
# set priors
z_prior ~ MvNormalMeanPrecision(zeros(latent_dim), diagm(ones(latent_dim)))
z_tmp   ~ BIFMHelper(z_prior) where { q = MeanField()}

# update last/previous hidden state
z_prev = z_tmp

# loop through observations
for i in 1:nr_samples

    # specify input as random variable
    u[i]   ~ MvNormalMeanPrecision(μu, Wu)

    # specify observation
    xt[i]  ~ BIFM(u[i], z_prev, z[i]) where { meta = BIFMMeta(A, B, C) }
    x[i]   ~ MvNormalMeanPrecision(xt[i], Wx)
    
    # update last/previous hidden state
    z_prev = z[i]

end
```
"""
struct BIFM end

@node BIFM Deterministic [ out, in, zprev, znext ]

@doc raw"""
The BIFMMeta structure contains all characterizing information of the BIFM node. In addition, it stores intermediate variables for efficient computations.
"""
mutable struct BIFMMeta{T}
    "Transition matrix A: converting the previous to the next latent state."
    A       :: Matrix{T}
    "Transition matrix B: converting the input to the latent state."
    B       :: Matrix{T}
    "Transition matrix C: converting the latent state to the output."
    C       :: Matrix{T}
    H       :: Union{Matrix{T}, Nothing}
    BHBt    :: Union{Matrix{T}, Nothing}
    ξz      :: Union{Vector{T}, Nothing}
    Λz      :: Union{Matrix{T}, Nothing}
    ξztilde :: Union{Vector{T}, Nothing}
    Λztilde :: Union{Matrix{T}, Nothing}
    μu      :: Union{Vector{T}, Nothing}
    Σu      :: Union{Matrix{T}, Nothing}
end

@doc raw"""
Initialization of the BIFMMeta object can be performed by calling
```julia
meta = BIFMMeta(A, B, C)
```
where `A`, `B` and `C` are the transition matrices in the model.
"""
function BIFMMeta(A::Array{T1, 2}, B::Array{T2, 2}, C::Array{T3, 2}) where { T1, T2, T3 }
    T = promote_type(T1, T2, T3)
    # check whether the dimensionality of transition matrices makes sense
    @assert size(A,1) == size(B,1)
    @assert size(A,1) == size(C,2)

    # return default metadata for BIFM node
    return BIFMMeta{T}(A, B, C, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end
@doc raw"""
Initialization of the BIFMMeta object can be performed by calling
```julia
meta = BIFMMeta(A, B, C, μu, Σu)
```
where `A`, `B` and `C` are the transition matrices in the model. `μu` and `Σu` are the mean vector and covariance matrix of the input. Importantly, in this setting we assume that these are known and do not change due to an external model.
"""
function BIFMMeta(A::Array{T1, 2}, B::Array{T2, 2}, C::Array{T3, 2}, μu::Array{T4, 1}, Σu::Array{T5, 2}) where { T1, T2, T3, T4, T5 }
    T = promote_type(T1, T2, T3, T4, T5)
    # check whether the dimensionality of transition matrices makes sense
    @assert size(A,1)  == size(B,1)
    @assert size(A,1)  == size(C,2)
    @assert size(μu,1) == size(Σu,1)
    @assert size(μu,1) == size(B,2)

    # return default Meta data for BIFM node
    return BIFMMeta{T}(A, B, C, nothing, nothing, nothing, nothing, nothing, nothing, μu, Σu)
end

getA(meta::BIFMMeta)                = meta.A
getB(meta::BIFMMeta)                = meta.B
getC(meta::BIFMMeta)                = meta.C
getH(meta::BIFMMeta)                = meta.H
getBHBt(meta::BIFMMeta)             = meta.BHBt
getξz(meta::BIFMMeta)               = meta.ξz
getΛz(meta::BIFMMeta)               = meta.Λz
getξztilde(meta::BIFMMeta)          = meta.ξztilde
getΛztilde(meta::BIFMMeta)          = meta.Λztilde
getμu(meta::BIFMMeta)               = meta.μu
getΣu(meta::BIFMMeta)               = meta.Σu

function setH!(meta::BIFMMeta, H)
    meta.H = H
end

function setBHBt!(meta::BIFMMeta, BHBt)
    meta.BHBt = BHBt
end

function setξztilde!(meta::BIFMMeta, ξztilde)
    meta.ξztilde = ξztilde
end

function setΛztilde!(meta::BIFMMeta, Λztilde)
    meta.Λztilde = Λztilde
end

function setξz!(meta::BIFMMeta, ξz)
    meta.ξz = ξz
end

function setΛz!(meta::BIFMMeta, Λz)
    meta.Λz = Λz
end

function setμu!(meta::BIFMMeta, μu)
    meta.μu = μu
end

function setΣu!(meta::BIFMMeta, Σu)
    meta.Σu = Σu
end

default_meta(::Type{ BIFM }) = error("BIFM node requires meta flag explicitly specified")