# distritbutions
@rule typeof(-)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily) = begin
    mout, vout = mean_var(m_out)
    min1, vin1 = mean_var(m_in1)
    return NormalMeanVariance(- mout + min1, vout + vin1)
end

@rule typeof(-)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily) = begin
    mout, vout = mean_cov(m_out)
    min1, vin1 = mean_cov(m_in1)
    return MvNormalMeanCovariance(- mout + min1, vout + vin1)
end

# PointMass
@rule typeof(-)(:in2, Marginalisation) (m_out::PointMass, m_in1::PointMass) = PointMass(- mean(m_out) + mean(m_in1))

@rule typeof(-)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalMeanPrecision) = begin
    min1, win1 = mean_precision(m_in1)
    return NormalMeanPrecision(- mean(m_out) + min1, win1)
end

@rule typeof(-)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return NormalMeanPrecision(- mout + mean(m_in1), wout)
end


@rule typeof(-)(:in2, Marginalisation) (m_out::PointMass, m_in1::MvNormalMeanPrecision) = begin
    min1, win1 = mean_precision(m_in1)
    return MvNormalMeanPrecision(- mean(m_out) + min1, win1)
end

@rule typeof(-)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return MvNormalMeanPrecision(- mout + mean(m_in1), wout)
end


@rule typeof(-)(:in2, Marginalisation) (m_out::PointMass, m_in1::UnivariateNormalDistributionsFamily) = begin
    min1, vin1 = mean_var(m_in1)
    return NormalMeanVariance(- mean(m_out) + min1, vin1)
end

@rule typeof(-)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass) = begin
    mout, vout = mean_var(m_out)
    return NormalMeanVariance(- mout + mean(m_in1), vout)
end


@rule typeof(-)(:in2, Marginalisation) (m_out::PointMass, m_in1::MultivariateNormalDistributionsFamily) = begin
    min1, vin1 = mean_cov(m_in1)
    return MvNormalMeanCovariance(- mean(m_out) + min1, vin1)
end

@rule typeof(-)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::PointMass) = begin
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(- mout + mean(m_in1), vout)
end

