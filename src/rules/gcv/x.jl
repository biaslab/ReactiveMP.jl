export rule

@rule GCV(:x, Marginalisation) (m_y::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * var(q_κ)
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

    return NormalMeanVariance(mean(m_y), var(m_y) + inv(A * B))
end

@rule GCV(:x, Marginalisation) (m_y::ContinuousUnivariateLogPdf,m_x::UnivariateNormalDistributionsFamily, q_z::Any, q_κ::Any, q_ω::Any) = begin
    m_yhat = @call_rule GCV(:y, Marginalisation) (m_x = m_x, q_z=q_z, q_κ=q_κ, q_ω=q_ω, meta = meta)
    approx_y = prod(ProdAnalytical(),m_yhat,m_y)
    
    mm_yhat, mw_yhat = mean(m_yhat), precision(m_yhat)
    qm_y, qw_y = mean(approx_y), precision(approx_y)
    qw_y = max(qw_y, mw_yhat) 

    v = inv(qw_y - mw_yhat)
    m = v*(qw_y*qm_y - mm_yhat*mw_yhat)

    return @call_rule GCV(:y, Marginalisation) (m_x = NormalMeanVariance(m, v), q_z=q_z, q_κ=q_κ, q_ω=q_ω, meta = meta)
end

@rule GCV(:x, Marginalisation) (q_y::Any, q_z::Any, q_κ::Any, q_ω::Any) = begin

    ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * var(q_κ)
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)

    return NormalMeanVariance(mean(q_y),  inv(A * B))
end