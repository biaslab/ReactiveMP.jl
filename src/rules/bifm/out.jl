
@rule BIFM(:out, Marginalisation) (m_in::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin

    # fetch information from meta data
    A       = getA(meta)
    B       = getB(meta)
    C       = getC(meta)
    H       = getH(meta)
    ξztilde = getξztilde(meta)
    BHBt    = getBHBt(meta)
    Λz      = getΛz(meta)

    # fetch statistics
    μ_in, Σ_in       = mean_cov(m_in)
    μ_zprev, Σ_zprev = mean_cov(m_zprev)

    # calculate intermediate quantities
    F = I - Λz * BHBt

    μ_ztilde = A * μ_zprev
    Σ_ztilde = A * Σ_zprev * A'

    μ_znext = F' * μ_ztilde + (B * (Σ_in * (B' * ξztilde))) + B * μ_in
    Σ_znext = F' * Σ_ztilde * F + BHBt

    # save required intermediate variables
    setΣu!(meta, Σ_in)

    # calculate outgoing message to output
    μ_out = C * μ_znext
    Σ_out = C * Σ_znext * C'

    # return outgoing marginal
    return MarginalDistribution(MvNormalMeanCovariance(μ_out, Σ_out))

end