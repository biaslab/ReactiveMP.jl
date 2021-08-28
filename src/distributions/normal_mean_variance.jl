export NormalMeanVariance

import StatsFuns: log2π, invsqrt2π

struct NormalMeanVariance{T <: Real} <: ContinuousUnivariateDistribution
    μ :: T
    v :: T
end

NormalMeanVariance(μ::Real, v::Real)       = NormalMeanVariance(promote(μ, v)...)
NormalMeanVariance(μ::Integer, v::Integer) = NormalMeanVariance(float(μ), float(v))
NormalMeanVariance(μ::T) where {T <: Real} = NormalMeanVariance(μ, one(T))
NormalMeanVariance()                       = NormalMeanVariance(0.0, 1.0)

Distributions.@distr_support NormalMeanVariance -Inf Inf

Distributions.support(dist::NormalMeanVariance) = Distributions.RealInterval(minimum(dist), maximum(dist))

weightedmean(dist::NormalMeanVariance) = precision(dist) * mean(dist)

function weightedmean_invcov(dist::NormalMeanVariance)
    w = invcov(dist)
    xi = w * mean(dist)
    return (xi, w)
end

Distributions.mean(dist::NormalMeanVariance)      = dist.μ
Distributions.median(dist::NormalMeanVariance)    = mean(dist)
Distributions.mode(dist::NormalMeanVariance)      = mean(dist)
Distributions.var(dist::NormalMeanVariance)       = dist.v
Distributions.std(dist::NormalMeanVariance)       = sqrt(var(dist))
Distributions.cov(dist::NormalMeanVariance)       = var(dist)
Distributions.invcov(dist::NormalMeanVariance)    = inv(cov(dist))
Distributions.entropy(dist::NormalMeanVariance)   = (1 + log2π + log(var(dist))) / 2

Distributions.pdf(dist::NormalMeanVariance, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) / 2cov(dist))) / std(dist)
Distributions.logpdf(dist::NormalMeanVariance, x::Real) = -(log2π + log(var(dist)) + abs2(x - mean(dist)) / var(dist)) / 2

Base.precision(dist::NormalMeanVariance{T}) where T = invcov(dist)
Base.eltype(::NormalMeanVariance{T})        where T = T

Base.convert(::Type{ NormalMeanVariance }, μ::Real, v::Real) = NormalMeanVariance(μ, v)
Base.convert(::Type{ NormalMeanVariance{T} }, μ::Real, v::Real) where { T <: Real } = NormalMeanVariance(convert(T, μ), convert(T, v))

vague(::Type{ <: NormalMeanVariance }) = NormalMeanVariance(0.0, huge)

prod_analytical_rule(::Type{ <: NormalMeanVariance }, ::Type{ <: NormalMeanVariance }) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdPreserveType, left::NormalMeanVariance, right::NormalMeanVariance)
    μ = (mean(left) * var(right) + mean(right) * var(left)) / (var(right) + var(left))
    v = (var(left) * var(right)) / (var(left) + var(right))
    return NormalMeanVariance(μ, v)
end

function Base.prod(::ProdAnalytical, left::NormalMeanVariance, right::NormalMeanVariance)
    xi = mean(left)/var(left) + mean(right)/var(right)
    w = 1/var(left) + 1/var(right)
    return NormalWeightedMeanPrecision(xi, w)
end