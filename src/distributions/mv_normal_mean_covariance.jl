export MvNormalMeanCovariance

import Distributions: logdetcov, distrname, sqmahal, sqmahal!, AbstractMvNormal
import LinearAlgebra: diag, Diagonal, dot
import Base: ndims, precision, length, size, prod

struct MvNormalMeanCovariance{ T <: Real, M <: AbstractVector{T}, P <: AbstractMatrix{T} } <: AbstractMvNormal
    μ :: M
    Σ :: P
end

function MvNormalMeanCovariance(μ::AbstractVector{ <: Real }, Σ::AbstractMatrix{ <: Real }) 
    T = promote_type(eltype(μ), eltype(Σ))
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

function MvNormalMeanCovariance(μ::AbstractVector{ <: Integer}, Σ::AbstractMatrix{ <: Integer }) 
    return MvNormalMeanCovariance(float.(μ), float.(Σ))
end

function MvNormalMeanCovariance(μ::AbstractVector, σ::AbstractVector)
    return MvNormalMeanCovariance(μ, matrix_from_diagonal(promote_type(eltype(μ), eltype(σ)), σ))
end

function MvNormalMeanCovariance(μ::AbstractVector{T}) where T
    return MvNormalMeanCovariance(μ, convert(AbstractArray{T}, ones(length(μ))))
end

Distributions.distrname(::MvNormalMeanCovariance) = "MvNormalMeanCovariance"

weightedmean(dist::MvNormalMeanCovariance) = invcov(dist) * mean(dist)

function weightedmean_invcov(dist::MvNormalMeanCovariance)
    W = invcov(dist)
    xi = W * mean(dist)
    return (xi, W)
end

Distributions.mean(dist::MvNormalMeanCovariance)      = dist.μ
Distributions.var(dist::MvNormalMeanCovariance)       = diag(cov(dist))
Distributions.cov(dist::MvNormalMeanCovariance)       = dist.Σ
Distributions.invcov(dist::MvNormalMeanCovariance)    = cholinv(dist.Σ)
Distributions.std(dist::MvNormalMeanCovariance)       = cholsqrt(cov(dist))
Distributions.logdetcov(dist::MvNormalMeanCovariance) = logdet(cov(dist))

Distributions.sqmahal(dist::MvNormalMeanCovariance, x::AbstractVector) = sqmahal!(similar(x), dist, x)

function Distributions.sqmahal!(r, dist::MvNormalMeanCovariance, x::AbstractVector)
    μ = mean(dist)
    for i in 1:length(r)
        @inbounds r[i] = μ[i] - x[i]
    end
    return xT_A_x(r, invcov(dist))
end

Base.eltype(::MvNormalMeanCovariance{T})     where T = T 
Base.precision(dist::MvNormalMeanCovariance)         = invcov(dist)
Base.length(dist::MvNormalMeanCovariance)            = length(mean(dist))
Base.ndims(dist::MvNormalMeanCovariance)             = length(dist)
Base.size(dist::MvNormalMeanCovariance)              = (length(dist), )


Base.convert(::Type{ <: MvNormalMeanCovariance }, μ::AbstractVector, Σ::AbstractMatrix) = MvNormalMeanCovariance(μ, Σ)

function Base.convert(::Type{ <: MvNormalMeanCovariance{T} }, μ::AbstractVector, Σ::AbstractMatrix) where { T <: Real }
    return MvNormalMeanCovariance(convert(AbstractArray{T}, μ), convert(AbstractArray{T}, Σ))
end

vague(::Type{ <: MvNormalMeanCovariance }, dims::Int) = MvNormalMeanCovariance(zeros(dims), fill(huge, dims))

prod_analytical_rule(::Type{ <: MvNormalMeanCovariance }, ::Type{ <: MvNormalMeanCovariance }) = ProdAnalyticalRuleAvailable()

function Base.prod(::ProdPreserveType, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    invcovleft  = invcov(left)
    invcovright = invcov(right)
    Σ = cholinv(invcovleft + invcovright)
    μ = Σ * (invcovleft * mean(left) + invcovright * mean(right))
    return MvNormalMeanCovariance(μ, Σ)
end

function Base.prod(::ProdAnalytical, left::MvNormalMeanCovariance, right::MvNormalMeanCovariance)
    xi_left, W_left = weightedmean_precision(left)
    xi_right, W_right = weightedmean_precision(right)
    return MvNormalWeightedMeanPrecision(xi_left + xi_right, W_left + W_right)
end


