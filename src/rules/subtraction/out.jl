# distritbutions
@rule typeof(-)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    min1, vin1 = mean_var(m_in1)
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(min1 - min2, vin1 + vin2)
end

@rule typeof(-)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    min1, vin1 = mean_cov(m_in1)
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(min1 - min2, vin1 + vin2)
end

# PointMass
@rule typeof(-)(:out, Marginalisation) (m_in1::PointMass, m_in2::PointMass) = PointMass(mean(m_in1) - mean(m_in2))


@rule typeof(-)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::PointMass) = begin
    min1, win1 = mean_precision(m_in1)
    return NormalMeanPrecision(min1 - mean(m_in2), win1)
end

@rule typeof(-)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return NormalMeanPrecision(mean(m_in1) - min2, win2)
end


@rule typeof(-)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::PointMass) = begin
    min1, win1 = mean_precision(m_in1)
    return MvNormalMeanPrecision(min1 - mean(m_in2), win1)
end

@rule typeof(-)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return MvNormalMeanPrecision(mean(m_in1) - min2, win2)
end


@rule typeof(-)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    min1, vin1 = mean_var(m_in1)
    return NormalMeanVariance(min1 - mean(m_in2), vin1)
end

@rule typeof(-)(:out, Marginalisation) (m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(mean(m_in1) - min2, vin2)
end


@rule typeof(-)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    min1, vin1 = mean_cov(m_in1)
    return MvNormalMeanCovariance(min1 - mean(m_in2), vin1)
end

@rule typeof(-)(:out, Marginalisation) (m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mean(m_in1) - min2, vin2)
end


