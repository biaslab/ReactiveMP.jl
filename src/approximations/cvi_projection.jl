export CVIProjection

export CVISamplingStrategy, FullSampling, MeanBased

"""
    CVISamplingStrategy

An abstract type representing the sampling strategy for the CVI projection method.
Concrete subtypes implement different approaches for generating samples used in 
approximating distributions.
"""
abstract type CVISamplingStrategy end

"""
    FullSampling <: CVISamplingStrategy
    FullSampling(samples::Int = 10)

A sampling strategy that uses multiple samples drawn from distributions.

# Arguments
- `samples::Int`: The number of samples to draw from each distribution. Default is 10.

# Example
```julia
# Use 100 samples for more accurate approximation
strategy = FullSampling(100)
```
"""
struct FullSampling <: CVISamplingStrategy
    samples::Int

    FullSampling(samples::Int = 10) = new(samples)
end

"""
    MeanBased <: CVISamplingStrategy

A sampling strategy that uses only the mean of the proposal distribution as a single sample.
"""
struct MeanBased <: CVISamplingStrategy end

"""
    ProposalDistributionContainer{PD}

A mutable wrapper for proposal distributions used in the CVI projection method.

The container allows the proposal distribution to be updated during inference
without recreating the entire approximation method structure.

# Fields
- `distribution::PD`: The wrapped proposal distribution, can be of any compatible type.
"""
mutable struct ProposalDistributionContainer{PD}
    distribution::PD
end

"""
    CVIProjection(; parameters...)

A structure representing the parameters for the Conjugate Variational Inference (CVI) projection method. 
This structure is a subtype of `AbstractApproximationMethod` and is used to configure the settings for CVI.

CVI approximates the posterior distribution by projecting it onto a family of distributions with a conjugate form.

# Requirements

The `CVIProjection` method requires the `ExponentialFamilyProjection` package to be installed and loaded
in the current environment with `using ExponentialFamilyProjection`.

# Parameters

- `rng::R`: The random number generator used for sampling. Default is `Random.MersenneTwister(42)`.
- `outsamples::S`: The number of samples used for approximating output message distributions. Default is `100`.
- `out_prjparams::OF`: The form parameter used to specify the target distribution family for the output message. 
   If `nothing` (default), the form will be inferred from the marginal form.
- `in_prjparams::IFS`: A NamedTuple-like object that specifies the target distribution family for each input edge.
   Keys should be of the form `:in_k` where `k` is the input edge index. If `nothing` (default), the forms
   will be inferred from the incoming messages.
- `proposal_distribution::PD`: The proposal distribution used for generating samples. If not provided or set to
   `nothing`, it will be inferred from incoming messages and automatically updated during iterations.
- `sampling_strategy::SS`: The strategy for approximating the logpdf:
  - `FullSampling(n)`: Uses `n` samples drawn from distributions (default: `n=10`). Provides more accurate
     approximation at the cost of increased computation time.
  - `MeanBased()`: Uses only the mean of each distribution as a single sample. Significantly faster but
     less accurate for non-linear nodes or complex distributions.

# Examples

```julia
# Standard CVI projection with default settings
method = CVIProjection()

# Fast approximation using mean-based sampling
method = CVIProjection(sampling_strategy = MeanBased())

# Custom proposal with increased sample count
using Distributions
proposal = FactorizedJoint((NormalMeanVariance(0.0, 1.0), NormalMeanVariance(0.0, 1.0)))
method = CVIProjection(
    proposal_distribution = ProposalDistributionContainer(proposal),
    sampling_strategy = FullSampling(1000)
)

# Specify projection family for the output message
method = CVIProjection(out_prjparams = NormalMeanPrecision)

# Specify projection family for input edges
method = CVIProjection(in_prjparams = (in_1 = NormalMeanVariance, in_2 = GammaMeanShape))
```

!!! note
    The `CVIProjection` method is an enhanced version of the deprecated `CVI`, offering better stability 
    and improved accuracy. Parameters and defaults may change as the implementation evolves.
"""
Base.@kwdef struct CVIProjection{R, S, OF, IFS, PD, SS} <: AbstractApproximationMethod
    rng::R = Random.MersenneTwister(42)
    outsamples::S = 100
    out_prjparams::OF = nothing
    in_prjparams::IFS = nothing
    proposal_distribution::PD = ProposalDistributionContainer{Any}(nothing)
    sampling_strategy::SS = FullSampling(10)
end

function get_kth_in_form(::CVIProjection{R, S, OF, Nothing}, ::Int) where {R, S, OF}
    return nothing
end

function get_kth_in_form(method::CVIProjection, k::Int)
    key = Symbol("in_$k")
    return get(method.in_prjparams, key, nothing)
end

# This method should only be invoked if a user did not install `ExponentialFamilyProjection`
# in the current Julia session
check_delta_node_compatibility(::Val{false}, ::CVIProjection) = error("CVI projection requires `using ExponentialFamilyProjection` in the current session.")
