export NormalMeanPrecision

import StatsFuns: log2π, invsqrt2π

struct NormalMeanPrecision{T <: Real} <: ContinuousUnivariateDistribution
    μ :: T
    w :: T
end

NormalMeanPrecision(μ::Real, w::Real)       = NormalMeanPrecision(promote(μ, w)...)
NormalMeanPrecision(μ::Integer, w::Integer) = NormalMeanPrecision(float(μ), float(w))
NormalMeanPrecision(μ::Real)                = NormalMeanPrecision(μ, one(μ))
NormalMeanPrecision()                       = NormalMeanPrecision(0.0, 1.0)

Distributions.@distr_support NormalMeanPrecision -Inf Inf

Distributions.support(dist::NormalMeanPrecision) = Distributions.RealInterval(minimum(dist), maximum(dist))

weightedmean(dist::NormalMeanPrecision) = precision(dist) * mean(dist)

Distributions.mean(dist::NormalMeanPrecision)    = dist.μ
Distributions.median(dist::NormalMeanPrecision)  = mean(dist)
Distributions.mode(dist::NormalMeanPrecision)    = mean(dist)
Distributions.var(dist::NormalMeanPrecision)     = inv(dist.w)
Distributions.std(dist::NormalMeanPrecision)     = sqrt(var(dist))
Distributions.cov(dist::NormalMeanPrecision)     = var(dist)
Distributions.invcov(dist::NormalMeanPrecision)  = dist.w
Distributions.entropy(dist::NormalMeanPrecision) = (1 + log2π - log(precision(dist))) / 2

Distributions.pdf(dist::NormalMeanPrecision, x::Real)    = (invsqrt2π * exp(-abs2(x - mean(dist)) * precision(dist) / 2)) * sqrt(precision(dist))
Distributions.logpdf(dist::NormalMeanPrecision, x::Real) = -(log2π - log(precision(dist)) + abs2(x - mean(dist)) * precision(dist)) / 2

Base.precision(dist::NormalMeanPrecision)     = invcov(dist)
Base.eltype(::NormalMeanPrecision{T}) where T = T

Base.convert(::Type{ NormalMeanPrecision }, μ::Real, w::Real) = NormalMeanPrecision(μ, w)
Base.convert(::Type{ NormalMeanPrecision{T} }, μ::Real, w::Real) where { T <: Real } = NormalMeanPrecision(convert(T, μ), convert(T, w))

vague(::Type{ <: NormalMeanPrecision }) = NormalMeanPrecision(0.0, tiny)

prod_analytical_rule(::Type{ <: NormalMeanPrecision }, ::Type{ <: NormalMeanPrecision }) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdPreserveType, left::NormalMeanPrecision, right::NormalMeanPrecision) 
    w = precision(left) + precision(right)
    μ = (mean(left) * precision(left) + mean(right) * precision(right)) / w
    return NormalMeanPrecision(μ, w)
end

function Base.prod(::ProdAnalytical, left::NormalMeanPrecision, right::NormalMeanPrecision) 
    w = precision(left) + precision(right)
    xi = mean(left)*precision(left) + mean(right)*precision(right)
    return NormalWeightedMeanPrecision(xi, w)
end