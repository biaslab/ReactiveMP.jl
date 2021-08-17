export marginalrule

@marginalrule GCV(:y_x) (m_y::UnivariateNormalDistributionsFamily, m_x::UnivariateNormalDistributionsFamily, q_z::Any, q_κ::Any, q_ω::Any) = begin

    ksi = mean(q_κ) ^ 2 * var(q_z) + mean(q_z) ^ 2 * var(q_κ) + var(q_z) * var(q_κ)
    A = exp(-mean(q_ω) + var(q_ω) / 2)
    B = exp(-mean(q_κ) * mean(q_z) + ksi / 2)
    W = [ precision(m_y) + A * B -A * B; -A * B precision(m_x) + A * B ]
    m = cholinv(W) * [ mean(m_y) * precision(m_y); mean(m_x) * precision(m_x) ]

    return MvNormalMeanPrecision(m, W)
end

@marginalrule GCV(:y_x) (m_y::ContinuousUnivariateLogPdf, m_x::UnivariateNormalDistributionsFamily, q_z::Any, q_κ::Any, q_ω::Any) = begin

    m_yhat = @call_rule GCV(:y, Marginalisation) (m_x = m_x, q_z=q_z, q_κ=q_κ, q_ω=q_ω, meta = meta)
    approx_y = prod(ProdAnalytical(),m_yhat,m_y)
    
    mm_yhat, mw_yhat = mean(m_yhat),precision(m_yhat)
    qm_y, qw_y = mean(approx_y),precision(approx_y)
    qw_y = max(qw_y, mw_yhat) + tiny 

    v = inv(qw_y - mw_yhat)
    m = v*(qw_y*qm_y - mm_yhat*mw_yhat)

    my = NormalMeanVariance(m, v)

    return @call_marginalrule GCV(:y_x) (m_y=my, m_x = m_x, q_z=q_z, q_κ=q_κ, q_ω=q_ω, meta = meta)
end

@marginalrule GCV(:y_x) (m_y::UnivariateNormalDistributionsFamily, m_x::ContinuousUnivariateLogPdf, q_z::Any, q_κ::Any, q_ω::Any) = begin

    return @call_marginalrule GCV(:y_x) (m_y=m_x, m_x = m_y, q_z=q_z, q_κ=q_κ, q_ω=q_ω, meta = meta)
end