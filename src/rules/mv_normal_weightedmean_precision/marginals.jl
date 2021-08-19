@marginalrule MvNormalWeightedMeanPrecision(:out_ξ_Λ) (m_out::MultivariateNormalDistributionsFamily, m_ξ::PointMass, m_Λ::PointMass, ) = begin 
    return (m_out = prod(ProdAnalytical(), MvNormalWeightedMeanPrecision(mean(m_ξ), mean(m_Λ)), m_out), m_ξ = m_ξ, m_Λ = m_Λ)
end