
@rule BIFM(:zprev, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin

    # fetch information from meta data
    A       = getA(meta)
    B       = getB(meta)
    C       = getC(meta)

    # fetch statistics from messages
    ξ_in, Λ_in          = weightedmean_precision(m_in)    
    ξ_out, Λ_out        = weightedmean_precision(m_out) 
    ξ_znext, Λ_znext    = weightedmean_precision(m_znext)

    # calculate intermediate quantities
    ξ_z = C' * ξ_out + ξ_znext
    Λ_z = C' * Λ_out * C + Λ_znext
    H = cholinv(Λ_in + B' * Λ_z * B)
    BHBt = B * H * B'
    ξ_ztilde = ξ_z + Λ_z * (B * (H * (-ξ_in - B' * ξ_z)))
    Λ_ztilde = Λ_z * (I - BHBt * Λ_z)

    # save required intermediate quantities
    setH!(meta, H)
    setξztilde!(meta, ξ_ztilde)
    setΛztilde!(meta, Λ_ztilde)
    setξz!(meta, ξ_z)
    setΛz!(meta, Λ_z)
    setBHBt!(meta, BHBt)

    # calculate outgoing message to zprev
    ξ_zprev = A' * ξ_ztilde
    Λ_zprev = A' * Λ_ztilde * A

    # return message
    return MvNormalWeightedMeanPrecision(ξ_zprev, Λ_zprev)

end