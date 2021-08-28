
@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    return (in1 = prod(ProdAnalytical(), NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out)), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out)), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanCovariance, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, Vout = mean_cov(m_out)
    return (in1 = prod(ProdAnalytical(), MvNormalMeanCovariance(mout - mean(m_in2), Vout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanPrecision, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, Wout = mean_precision(m_out)
    return (in1 = prod(ProdAnalytical(), MvNormalMeanPrecision(mout - mean(m_in2), Wout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalWeightedMeanPrecision, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    xiout, Wout = weightedmean_precision(m_out)
    return (in1 = prod(ProdAnalytical(), MvNormalWeightedMeanPrecision(xiout - Wout*mean(m_in2), Wout), m_in1), in2 = m_in2)
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanCovariance, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, Vout = mean_cov(m_out)
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), MvNormalMeanCovariance(mout - mean(m_in1), Vout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalMeanPrecision, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, Wout = mean_precision(m_out)
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), MvNormalMeanPrecision(mout - mean(m_in1), Wout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::MvNormalWeightedMeanPrecision, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    xiout, Wout = weightedmean_precision(m_out)
    return (in1 = m_in1, in2 = prod(ProdAnalytical(), MvNormalWeightedMeanPrecision(xiout - Wout * mean(m_in1), Wout), m_in2))
end

@marginalrule typeof(+)(:in1_in2) (m_out::NormalDistributionsFamily, m_in1::NormalDistributionsFamily, m_in2::NormalDistributionsFamily) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_in1, W_in1 = weightedmean_precision(m_in1)
    xi_in2, W_in2 = weightedmean_precision(m_in2)
    
    return MvNormalWeightedMeanPrecision([ xi_in1 + xi_out; xi_in2 + xi_out ], [ W_in1 + W_out W_out; W_out W_in2 + W_out])
end