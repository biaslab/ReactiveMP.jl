export NormalWeightedMeanPrecision

import StatsFuns: log2π, invsqrt2π

struct NormalWeightedMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution
    xi :: T # Weighted mean: xi = w * μ
    w  :: T
end

NormalWeightedMeanPrecision(xi::Real, w::Real)       = NormalWeightedMeanPrecision(promote(xi, w)...)
NormalWeightedMeanPrecision(xi::Integer, w::Integer) = NormalWeightedMeanPrecision(float(xi), float(w))
NormalWeightedMeanPrecision(xi::Real)                = NormalWeightedMeanPrecision(xi, one(xi))
NormalWeightedMeanPrecision()                        = NormalWeightedMeanPrecision(0.0, 1.0)

Distributions.@distr_support NormalWeightedMeanPrecision -Inf Inf

Distributions.support(dist::NormalWeightedMeanPrecision) = Distributions.RealInterval(minimum(dist), maximum(dist))

weightedmean(dist::NormalWeightedMeanPrecision) = dist.xi

function mean_cov(dist::NormalWeightedMeanPrecision)
    v = cov(dist)
    μ = v * weightedmean(dist)
    return (μ, v)
end

Distributions.mean(dist::NormalWeightedMeanPrecision)    = var(dist) * weightedmean(dist)
Distributions.median(dist::NormalWeightedMeanPrecision)  = mean(dist)
Distributions.mode(dist::NormalWeightedMeanPrecision)    = mean(dist)
Distributions.var(dist::NormalWeightedMeanPrecision)     = inv(dist.w)
Distributions.std(dist::NormalWeightedMeanPrecision)     = sqrt(var(dist))
Distributions.cov(dist::NormalWeightedMeanPrecision)     = var(dist)
Distributions.invcov(dist::NormalWeightedMeanPrecision)  = dist.w
Distributions.entropy(dist::NormalWeightedMeanPrecision) = (1 + log2π - log(precision(dist))) / 2

Distributions.pdf(dist::NormalWeightedMeanPrecision, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
Distributions.logpdf(dist::NormalWeightedMeanPrecision, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::NormalWeightedMeanPrecision)     = invcov(dist)
Base.eltype(::NormalWeightedMeanPrecision{T}) where T = T

Base.convert(::Type{ NormalWeightedMeanPrecision }, xi::Real, w::Real) = NormalWeightedMeanPrecision(xi, w)
Base.convert(::Type{ NormalWeightedMeanPrecision{T} }, xi::Real, w::Real) where { T <: Real } = NormalWeightedMeanPrecision(convert(T, xi), convert(T, w))

vague(::Type{ <: NormalWeightedMeanPrecision }) = NormalWeightedMeanPrecision(0.0, tiny)

prod_analytical_rule(::Type{ <: NormalWeightedMeanPrecision }, ::Type{ <: NormalWeightedMeanPrecision }) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdPreserveType, left::NormalWeightedMeanPrecision, right::NormalWeightedMeanPrecision) 
    xi = weightedmean(left) + weightedmean(right)
    w  = precision(left) + precision(right)
    return NormalWeightedMeanPrecision(xi, w)
end

function Base.prod(::ProdAnalytical, left::NormalWeightedMeanPrecision, right::NormalWeightedMeanPrecision) 
    xi = weightedmean(left) + weightedmean(right)
    w  = precision(left) + precision(right)
    return NormalWeightedMeanPrecision(xi, w)
end