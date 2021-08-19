@marginalrule BIFM(:in_zprev_znext) (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta{Float64}) = begin 
    # Note: this rules is equal to the rule for the :in_zprev marginal. However, it yield identical performance with respect to the RTS smoother (aside from the additional prior on the last z)

    # fetch statistics from meta data
    A = getA(meta)
    B = getB(meta)
    C = getC(meta)
    ξ_ztilde = getξztilde(meta)
    Λ_ztilde = getΛztilde(meta)

    # # extract parameters from messages
    ξ_zprev_marginal, Λ_zprev_marginal = weightedmean_precision(m_zprev)
    ξ_in, Λ_in       = weightedmean_precision(m_in)
    ξ_out, Λ_out     = weightedmean_precision(m_out)

    # calculate outgoing message of zprev
    ξ_zprev_message = A' * ξ_ztilde
    Λ_zprev_message = A' * Λ_ztilde * A

    # calculate message towards znext from y
    dist1 = MvNormalWeightedMeanPrecision(C'*ξ_out, C'*Λ_out*C)

    # calculate message from z towards the addition node
    dist2 = prod(ProdAnalytical(), dist1, m_znext)
    ξ2, Λ2 = weightedmean_precision(dist2)

    # calculate joint message from the addition node
    BA = hcat(B, A)'
    ξ3 = BA*ξ2
    Λ3 = BA*Λ2*BA'

    # create a joint message of the input messages
    ξ4 = vcat(ξ_in, (ξ_zprev_marginal - ξ_zprev_message))
    Λ4 = cat(Λ_in, (Λ_zprev_marginal - Λ_zprev_message); dims=(1,2))

    # return joint marginal
    return prod(ProdAnalytical(), MvNormalWeightedMeanPrecision(ξ3, Λ3), MvNormalWeightedMeanPrecision(ξ4, Λ4))

end