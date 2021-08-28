# distritbutions
@rule typeof(-)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    mout, vout = mean_var(m_out)
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(mout + min2, vout + vin2)
end

@rule typeof(-)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, vout = mean_cov(m_out)
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mout + min2, vout + vin2)
end

# PointMass
@rule typeof(-)(:in1, Marginalisation) (m_out::PointMass, m_in2::PointMass) = PointMass(mean(m_out) + mean(m_in2))

@rule typeof(-)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return NormalMeanPrecision(mean(m_out) + min2, win2)
end

@rule typeof(-)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return NormalMeanPrecision(mout + mean(m_in2), wout)
end


@rule typeof(-)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return MvNormalMeanPrecision(mean(m_out) + min2, win2)
end

@rule typeof(-)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return MvNormalMeanPrecision(mout + mean(m_in2), wout)
end


@rule typeof(-)(:in1, Marginalisation) (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(mean(m_out) + min2, vin2)
end

@rule typeof(-)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, vout = mean_var(m_out)
    return NormalMeanVariance(mout + mean(m_in2), vout)
end


@rule typeof(-)(:in1, Marginalisation) (m_out::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mean(m_out) + min2, vin2)
end

@rule typeof(-)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(mout + mean(m_in2), vout)
end

