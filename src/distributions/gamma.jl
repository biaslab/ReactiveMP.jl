export Gamma, GammaShapeScale, GammaDistributionsFamily, GammaNaturalParameters

import SpecialFunctions: loggamma, digamma
import Distributions: Gamma, shape, scale, cov
import StatsFuns: log2π

const GammaShapeScale             = Gamma
const GammaDistributionsFamily{T} = Union{GammaShapeScale{T}, GammaShapeRate{T}}

Distributions.cov(dist::GammaDistributionsFamily) = var(dist)

function mean(::typeof(log), dist::GammaShapeScale)
    k, θ = params(dist)
    return digamma(k) + log(θ)
end

function mean(::typeof(loggamma), dist::GammaShapeScale)
    k, θ = params(dist)
    return 0.5 * (log2π - (digamma(k) + log(θ))) + mean(dist) * (-1 + digamma(k + 1) + log(θ))
end

function mean(::typeof(xtlog), dist::GammaShapeScale)
    k, θ = params(dist)
    return mean(dist) * (digamma(k + 1) + log(θ))
end

vague(::Type{<:GammaShapeScale}) = GammaShapeScale(1.0, huge)

convert_paramfloattype(::Type{T}, distribution::GammaShapeScale) where {T} = GammaShapeScale(convert.(T, params(distribution))...; check_args = false)
convert_paramfloattype(::Type{T}, distribution::GammaShapeRate) where {T} = GammaShapeRate(convert.(T, params(distribution))...) # our implementation does not check_args anyway

prod_analytical_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeScale}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::GammaShapeScale, right::GammaShapeScale)
    T = promote_paramfloattype(left, right)
    return GammaShapeScale(shape(left) + shape(right) - one(T), (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

# Conversion to shape - scale parametrisation

function Base.convert(::Type{GammaShapeScale{T}}, dist::GammaDistributionsFamily) where {T}
    return GammaShapeScale(convert(T, shape(dist)), convert(T, scale(dist)))
end

function Base.convert(::Type{GammaShapeScale}, dist::GammaDistributionsFamily{T}) where {T}
    return convert(GammaShapeScale{T}, dist)
end

# Conversion to shape - rate parametrisation

function Base.convert(::Type{GammaShapeRate{T}}, dist::GammaDistributionsFamily) where {T}
    return GammaShapeRate(convert(T, shape(dist)), convert(T, rate(dist)))
end

function Base.convert(::Type{GammaShapeRate}, dist::GammaDistributionsFamily{T}) where {T}
    return convert(GammaShapeRate{T}, dist)
end

# Extensions of prod methods

prod_analytical_rule(::Type{<:GammaShapeRate}, ::Type{<:GammaShapeScale}) = ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type{<:GammaShapeScale}, ::Type{<:GammaShapeRate}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::GammaShapeRate, right::GammaShapeScale)
    T = promote_samplefloattype(left, right)
    return GammaShapeRate(shape(left) + shape(right) - one(T), rate(left) + rate(right))
end

function prod(::ProdAnalytical, left::GammaShapeScale, right::GammaShapeRate)
    T = promote_samplefloattype(left, right)
    return GammaShapeScale(shape(left) + shape(right) - one(T), (scale(left) * scale(right)) / (scale(left) + scale(right)))
end

function compute_logscale(new_dist::GammaDistributionsFamily, left_dist::GammaDistributionsFamily, right_dist::GammaDistributionsFamily)
    ay, by = shape(new_dist), rate(new_dist)
    ax, bx = shape(left_dist), rate(left_dist)
    az, bz = shape(right_dist), rate(right_dist)
    return loggamma(ay) - loggamma(ax) - loggamma(az) + ax * log(bx) + az * log(bz) - ay * log(by)
end

prod_analytical_rule(::Type{<:Truncated{<:Normal}}, ::Type{<:GammaDistributionsFamily}) = ProdAnalyticalRuleAvailable()
prod_analytical_rule(::Type{<:GammaDistributionsFamily}, ::Type{<:Truncated{<:Normal}}) = ProdAnalyticalRuleAvailable()

prod(::ProdAnalytical, left::GammaDistributionsFamily, right::Truncated{<:Normal}) = prod(ProdAnalytical(), right, left)

function prod(::ProdAnalytical, left::Truncated{<:Normal}, right::GammaDistributionsFamily)
    @assert (left.lower ≈ zero(left.lower) && isinf(left.upper)) "Truncated{Normal} * Gamma only implemented for Truncated{Normal}(0, Inf)"

    samples = rand(MersenneTwister(123), left, 1000)
    zeronum = zero(eltype(samples))

    sx, xlogx, tw = mapreduce(.+, samples; init = (zeronum, zeronum, zeronum)) do sample
        w = pdf(right, sample)
        return (w * sample, w * log(sample), w)
    end

    statistics = Distributions.GammaStats(sx, xlogx, tw)
    fit = Distributions.fit_mle(Gamma, statistics, alpha0 = shape(right))

    return convert(typeof(right), fit)
end

## Friendly functions

function logpdf_sample_friendly(dist::GammaDistributionsFamily)
    friendly = convert(GammaShapeScale, dist)
    return (friendly, friendly)
end

## Natural parameters for the Gamma family of distributions

struct GammaNaturalParameters{T <: Real} <: NaturalParameters
    a::T
    b::T
end

GammaNaturalParameters(a::Real, b::Real)       = GammaNaturalParameters(promote(a, b)...)
GammaNaturalParameters(a::Integer, b::Integer) = GammaNaturalParameters(float(a), float(b))

function GammaNaturalParameters(vec::AbstractVector)
    @assert length(vec) === 2 "`GammaNaturalParameters` must accept a vector of length `2`."
    return GammaNaturalParameters(vec[1], vec[2])
end

Base.convert(::Type{GammaNaturalParameters}, a::Real, b::Real) = convert(GammaNaturalParameters{promote_type(typeof(a), typeof(b))}, a, b)

Base.convert(::Type{GammaNaturalParameters{T}}, a::Real, b::Real) where {T} = GammaNaturalParameters(convert(T, a), convert(T, b))

Base.convert(::Type{GammaNaturalParameters}, vec::AbstractVector) = convert(GammaNaturalParameters{eltype(vec)}, vec)

Base.convert(::Type{GammaNaturalParameters{T}}, vec::AbstractVector) where {T} = GammaNaturalParameters(convert(AbstractVector{T}, vec))

function Base.:(==)(left::GammaNaturalParameters, right::GammaNaturalParameters)
    return left.a == right.a && left.b == right.b
end

as_naturalparams(::Type{T}, args...) where {T <: GammaNaturalParameters} = convert(GammaNaturalParameters, args...)

function Base.convert(::Type{Distribution}, η::GammaNaturalParameters)
    return GammaShapeRate(η.a + 1, -η.b)
end

naturalparams(dist::GammaDistributionsFamily) = GammaNaturalParameters(shape(dist) - 1, -rate(dist))

# Natural parameters to standard dist. type

function Base.vec(p::GammaNaturalParameters)
    return [p.a, p.b]
end

function Base.:+(left::GammaNaturalParameters, right::GammaNaturalParameters)
    return GammaNaturalParameters(left.a + right.a, left.b + right.b)
end

function Base.:-(left::GammaNaturalParameters, right::GammaNaturalParameters)
    return GammaNaturalParameters(left.a - right.a, left.b - right.b)
end

function lognormalizer(η::GammaNaturalParameters)
    return loggamma(η.a + 1) - (η.a + 1) * log(-η.b)
end

function Distributions.logpdf(η::GammaNaturalParameters, x)
    return log(x) * η.a + x * η.b - lognormalizer(η)
end

function isproper(params::GammaNaturalParameters)
    return (params.a >= tiny - 1) && (-params.b >= tiny)
end
