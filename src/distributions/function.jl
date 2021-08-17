export ContinuousUnivariateLogPdf

using Distributions

import DomainSets
import DomainIntegrals
import Base: isapprox

"""
    ContinuousUnivariateLogPdf{ D <: DomainSets.Domain, F } <: ContinuousUnivariateDistribution

Generic continuous univariate distribution in a form of domain specification and logpdf function. Can be used in cases where no 
known analytical distribution available. 

# Arguments 
- `domain`: domain specificatiom from `DomainSets.jl` package
- `logpdf`: callable object that represents a `logpdf` of a distribution. Does not necessarily normalised.

```julia 
fdist = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> -x^2)
```
"""
struct ContinuousUnivariateLogPdf{ D <: DomainSets.Domain, F } <: ContinuousUnivariateDistribution
    domain :: D
    logpdf :: F
end

ContinuousUnivariateLogPdf(f::Function) = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), f)

(dist::ContinuousUnivariateLogPdf)(x::Real)                      = logpdf(dist, x)
(dist::ContinuousUnivariateLogPdf)(x::AbstractVector{ <: Real }) = logpdf(dist, x)

Distributions.support(dist::ContinuousUnivariateLogPdf) = Distributions.RealInterval(DomainSets.infimum(dist.domain), DomainSets.supremum(dist.domain))

function Distributions.mean(dist::ContinuousUnivariateLogPdf)    
    supp = Distributions.support(dist)

    if supp == RealInterval{Float64}(-Inf,Inf)
        cubat = GaussHermiteCubature(31)
        m,v = approximate_meancov(cubat,x -> pdf(dist,x),0.0,1.0)
    else
        error("mean() is not defined for `ContinuousUnivariateLogPdf`.")
    end

    return m
end

function Distributions.var(dist::ContinuousUnivariateLogPdf)    
    supp = Distributions.support(dist)

    if supp == RealInterval{Float64}(-Inf,Inf)
        cubat = GaussHermiteCubature(31)
        m,v = approximate_meancov(cubat,x -> pdf(dist,x),0.0,1.0)
    else
        error("mean() is not defined for `ContinuousUnivariateLogPdf`.")
    end

    return v
end
Distributions.precision(dist::ContinuousUnivariateLogPdf) = inv(var(dist))

Distributions.median(dist::ContinuousUnivariateLogPdf)  = error("median() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.mode(dist::ContinuousUnivariateLogPdf)    = error("mode() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.var(dist::ContinuousUnivariateLogPdf)     = error("var() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.std(dist::ContinuousUnivariateLogPdf)     = error("std() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.cov(dist::ContinuousUnivariateLogPdf)     = error("cov() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.invcov(dist::ContinuousUnivariateLogPdf)  = error("invcov() is not defined for `ContinuousUnivariateLogPdf`.")
Distributions.entropy(dist::ContinuousUnivariateLogPdf) = error("entropy() is not defined for `ContinuousUnivariateLogPdf`.")

# We don't expect neither `pdf` nor `logpdf` to be normalised
Distributions.pdf(dist::ContinuousUnivariateLogPdf, x::Real)    = exp(logpdf(dist, x))

function Distributions.logpdf(dist::ContinuousUnivariateLogPdf, x::Real) 
    @assert x ∈ dist.domain "x = $(x) does not belong to the domain of $dist"
    return dist.logpdf(x)
end

# These are fallbacks for various optimisation packages which may pass arguments as vectors
function Distributions.pdf(dist::ContinuousUnivariateLogPdf, x::AbstractVector{ <: Real })
    @assert length(x) === 1 "`ContinuousUnivariateLogPdf` expects either float or a vector of a single float as an input for the `pdf` function."
    return exp(logpdf(dist, first(x)))
end

function Distributions.logpdf(dist::ContinuousUnivariateLogPdf, x::AbstractVector{ <: Real }) 
    @assert length(x) === 1 "`ContinuousUnivariateLogPdf` expects either float or a vector of a single float as an input for the `logpdf` function."
    return logpdf(dist, first(x))
end

Base.precision(dist::ContinuousUnivariateLogPdf) = error("precision() is not defined for `ContinuousUnivariateLogPdf`.")

Base.convert(::Type{ ContinuousUnivariateLogPdf }, domain::D, logpdf::F) where { D <: DomainSets.Domain, F } = ContinuousUnivariateLogPdf{D, F}(domain, logpdf)

convert_eltype(::Type{ ContinuousUnivariateLogPdf }, ::Type{ T }, dist::ContinuousUnivariateLogPdf) where { T <: Real } = convert(ContinuousUnivariateLogPdf, dist.domain, dist.logpdf)

vague(::Type{ <: ContinuousUnivariateLogPdf }) = ContinuousUnivariateLogPdf(DomainSets.FullSpace(), (x) -> 1.0)

prod_analytical_rule(::Type{ <: ContinuousUnivariateLogPdf }, ::Type{ <: ContinuousUnivariateLogPdf }) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf)
    @assert left.domain == right.domain "Different domain types in product of generic `ContinuousUnivariateLogPdf` distributions. Left domain is $(left.domain), right is $(right.domain)."
    plogpdf = let left = left, right = right
        (x) -> logpdf(left, x) + logpdf(right, x)
    end
    return ContinuousUnivariateLogPdf(left.domain, plogpdf)
end

# This method is inaccurate and relies on various approximation methods, which may fail in different scenarious
# Current implementation of `isapprox` method supports only FullSpace and HalfLine domains with limited accuracy
function Base.isapprox(left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf; kwargs...)
    if left.domain !== right.domain
        return false
    end
    return culogpdf__isapprox(left.domain, left, right; kwargs...)
end

# https://en.wikipedia.org/wiki/Gauss–Hermite_quadrature
function culogpdf__isapprox(domain::DomainSets.FullSpace, left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf; kwargs...)
    return isapprox(zero(eltype(domain)), DomainIntegrals.integral(Q_GaussHermite(32), (x) -> exp(x ^ 2) * abs(left(x) - right(x))); kwargs...)
end

# https://en.wikipedia.org/wiki/Gauss–Laguerre_quadrature
function culogpdf__isapprox(domain::DomainSets.HalfLine, left::ContinuousUnivariateLogPdf, right::ContinuousUnivariateLogPdf; kwargs...)
    return isapprox(zero(eltype(domain)), DomainIntegrals.integral(Q_GaussLaguerre(32), (x) -> exp(x) * abs(left(x) - right(x))); kwargs...)
end

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but it does not really mean that objects are different
function is_typeof_equal(left::ContinuousUnivariateLogPdf{D, F1}, right::ContinuousUnivariateLogPdf{D, F2}) where { D, F1 <: Function, F2 <: Function }
    return true
end