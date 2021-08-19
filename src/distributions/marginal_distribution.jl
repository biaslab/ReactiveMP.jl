export MarginalDistribution

"""
The MarginalDistribution is a wrapper around a distribution. By passing it as a message along an edge of the graph the corresponding marginal is calculated as the distribution of the MarginalDistribution. In a sense, the MarginalDistribution overwrites any distribution for calculating the marginal, such as is the case with the BIFM node/smoother.
"""
struct MarginalDistribution{T <: Distribution}
    dist :: T
end
getdist(dist::MarginalDistribution) = dist.dist

# apply functions on containing distribution
mean(dist::MarginalDistribution)                    = mean(dist.dist)
median(dist::MarginalDistribution)                  = median(dist.dist)
mode(dist::MarginalDistribution)                    = mode(dist.dist)
shape(dist::MarginalDistribution)                   = shape(dist.dist)
scale(dist::MarginalDistribution)                   = scale(dist.dist)
rate(dist::MarginalDistribution)                    = rate(dist.dist)
var(dist::MarginalDistribution)                     = var(dist.dist)
std(dist::MarginalDistribution)                     = std(dist.dist)
cov(dist::MarginalDistribution)                     = cov(dist.dist)
invcov(dist::MarginalDistribution)                  = invcov(dist.dist)
entropy(dist::MarginalDistribution)                 = entropy(dist.dist)
pdf(dist::MarginalDistribution)                     = pdf(dist.dist)
logpdf(dist::MarginalDistribution)                  = logpdf(dist.dist)
logdetcov(dist::MarginalDistribution)               = logdetcov(dist.dist)
mean_cov(dist::MarginalDistribution)                = mean_cov(dist.dist)
mean_invcov(dist::MarginalDistribution)             = mean_invcov(dist.dist)
mean_precision(dist::MarginalDistribution)          = mean_precision(dist.dist)
weightedmean_cov(dist::MarginalDistribution)        = weightedmean_cov(dist.dist)
weightedmean_invcov(dist::MarginalDistribution)     = weightedmean_invcov(dist.dist)
weightedmean_precision(dist::MarginalDistribution)  = weightedmean_precision(dist.dist)
weightedmean(dist::MarginalDistribution)            = weightedmean(dist.dist)
probvec(dist::MarginalDistribution)                 = probvec(dist.dist)
logmean(dist::MarginalDistribution)                 = logmean(dist.dist)
meanlogmean(dist::MarginalDistribution)             = meanlogmean(dist.dist)
inversemean(dist::MarginalDistribution)             = inversemean(dist.dist)
mirroredlogmean(dist::MarginalDistribution)         = mirroredlogmean(dist.dist)
loggammamean(dist::MarginalDistribution)            = loggammamean(dist.dist)

custom_isapprox(dist1::MarginalDistribution, dist2::MarginalDistribution; kwargs...)  = custom_isapprox(dist1.dist, dist2.dist; kwargs...)
custom_isapprox(dist1::Any, dist2::MarginalDistribution; kwargs...)                   = custom_isapprox(dist1, dist2.dist; kwargs...)
custom_isapprox(dist1::MarginalDistribution, dist2::Any; kwargs...)                   = custom_isapprox(dist1.dist, dist2; kwargs...)

# product of distribution message with marginal message returns the marginal directly
function prod(::ProdAnalytical, left::MarginalDistribution, right::Distribution{F, S}) where { F <: VariateForm, S <: ValueSupport }
    return getdist(left)
end

function prod(::ProdAnalytical, left::Distribution{F, S}, right::MarginalDistribution) where { F <: VariateForm, S <: ValueSupport }
    return getdist(right)
end
