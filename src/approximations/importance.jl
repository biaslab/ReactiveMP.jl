export ImportanceSamplingApproximation

using Random
using StatsBase

"""
    ImportanceSamplingApproximation

This structure stores all information needed to perform an importance sampling procedure and provides
convenient functions to generate samples and weights to approximate expectations

# Fields
- `rng`: random number generator objects
- `nsamples`: number of samples generated by default
"""
struct ImportanceSamplingApproximation{T, R} <: AbstractFormConstraint
    rng        :: R
    nsamples   :: Int
    bsamples   :: Vector{T}
    bweights   :: Vector{T}
    resampling :: Bool
    rsamples   :: Vector{T}
end

function ImportanceSamplingApproximation(nsamples::Int; resampling::Bool = true)
    return ImportanceSamplingApproximation(Float64, Random.GLOBAL_RNG, nsamples; resampling = resampling)
end

function ImportanceSamplingApproximation(rng::R, nsamples::Int; resampling::Bool = true) where {R}
    return ImportanceSamplingApproximation(Float64, rng, nsamples; resampling = resampling)
end

function ImportanceSamplingApproximation(::Type{T}, rng::R, nsamples::Int; resampling::Bool = true) where {T, R}
    return ImportanceSamplingApproximation{T, R}(
        rng,
        nsamples,
        Vector{T}(undef, nsamples),
        Vector{T}(undef, nsamples),
        resampling,
        Vector{T}(undef, nsamples)
    )
end

getsamples(approximation::ImportanceSamplingApproximation, distribution)           = getsamples(approximation, distribution, approximation.nsamples)
getsamples(approximation::ImportanceSamplingApproximation, distribution, nsamples) = rand(approximation.rng, distribution, nsamples)

function approximate_meancov(approximation::ImportanceSamplingApproximation, g::Function, distribution)

    # We use preallocated arrays to sample and compute transformed samples and weightd
    rand!(approximation.rng, distribution, approximation.bsamples)
    map!(g, approximation.bweights, approximation.bsamples)

    if approximation.resampling
        n_eff = 1 / sum(Base.Generator(abs2, approximation.bweights))

        # Here we assume that lengths of bweights ans bsamples vectors are the same
        N = length(approximation.bweights)

        if n_eff < N / 10
            # We use rsamples as a temporary buffer here
            copyto!(approximation.rsamples, 1, approximation.bsamples, 1, N)
            # Here rsamples are equal to bsamples, but during sampling bsamples will be overwritten
            sample!(approximation.rng, approximation.rsamples, Weights(approximation.bweights), approximation.bsamples)
            fill!(approximation.bweights, 1 / N)
        end
    end

    normalization = sum(approximation.bweights)

    if iszero(normalization)
        return mean(distribution), var(distribution)
    end

    map!(Base.Fix2(/, normalization), approximation.bweights, approximation.bweights)

    m = mapreduce(prod, +, zip(approximation.bweights, approximation.bsamples))

    _v = let m = m
        (r) -> r[1] * (r[2] - m)^2
    end

    v = mapreduce(_v, +, zip(approximation.bweights, approximation.bsamples))

    if isnan(m) || isnan(v) || isinf(m) || iszero(v)
        return mean(distribution), var(distribution)
    end

    return m, v
end

is_point_mass_form_constraint(::ImportanceSamplingApproximation) = false
default_form_check_strategy(::ImportanceSamplingApproximation)   = FormConstraintCheckEach()
default_prod_constraint(::ImportanceSamplingApproximation)       = ProdGeneric()

function constrain_form(approximation::ImportanceSamplingApproximation, distribution::DistProduct)
    m, v = approximate_meancov(approximation, x -> pdf(getright(distribution), x), getleft(distribution))

    a = m^2 / v
    b = m / v
    return GammaShapeRate(a, b)
end

constrain_form(::ImportanceSamplingApproximation, distribution::Any) = distribution

make_form_constraint(::Type{ImportanceSamplingApproximation}, args...; kwargs...) =
    ImportanceSamplingApproximation(args..., kwargs...)
