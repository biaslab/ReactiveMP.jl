
@rule BIFM(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin

    # fetch information of meta data
    A                   = getA(meta)
    B                   = getB(meta)
    μu                  = getμu(meta)
    Σu                  = getΣu(meta)
    ξztilde             = getξztilde(meta)
    Λztilde             = getΛztilde(meta)

    # fetch statistics from messages
    μ_zprev, Σ_zprev    = mean_cov(m_zprev)

    # calculate intermediate variables (with dual parameterization)
    # see Wadehn 2016 - On sparsity by NUV-EM ...
    ξtildex = Λztilde * (A * μ_zprev) - ξztilde
    Λtildex = Λztilde - Λztilde * A * Σ_zprev * A' * Λztilde
    tmp = B * Σu

    # calculate marginals of input WRONG
    μ_in = μu - (Σu * (B' * ξtildex))
    Σ_in = Σu - tmp' * Λtildex * tmp

    # return input marginal
    return MarginalDistribution(MvNormalMeanCovariance(μ_in, Σ_in))

end