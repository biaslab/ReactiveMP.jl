
@rule Probit(:in, Marginalisation) (m_out::Union{PointMass, Bernoulli}, ) = begin
    
    # extract parameters
    p = mean(m_out)

    # calculate outward message
    f = (z) -> log(1 - p + (2*p - 1)*normcdf(z))

    # return message
    return ContinuousUnivariateLogPdf(f)

end

@rule Probit(:in, MomentMatching) (m_out::Union{PointMass, Bernoulli}, m_in::UnivariateNormalDistributionsFamily) = begin

    # extract parameters
    mz, vz = mean_cov(m_in)
    p = mean(m_out)
    @assert p >= zero(p) && p <= one(p) "The Probit node only accepts messages on its output with values between 0 and 1."

    # calculate auxiliary variables
    γ = mz/sqrt(1+vz)

    # calculate unnormalized moments of g
    umom1_g = normcdf(γ)*mz + vz*normpdf(γ)/sqrt(1+vz)
    umom2_g = 2*mz*umom1_g + (vz - mz^2)*normcdf(γ) - vz^2*γ*normpdf(γ)/(1+vz)

    # calculate moments of posterior
    mom0_pz = 1 - p + (2*p - 1)*normcdf(γ)
    mom1_pz = 1/mom0_pz*((1-p)*mz + (2*p-1)*umom1_g)
    mom2_pz = 1/mom0_pz*((1-p)*(vz + mz^2) + (2*p-1)*umom2_g)

    # calculate parameters of posterior
    mpz = mom1_pz
    vpz = mom2_pz - mom1_pz^2
    vpz = min(max(vpz, tiny), vz - tiny) # ensure variance of marginal is not larger than the variance of the cavity distribution.

    # calculate parameters of outgoing message
    wz_out = 1/vpz - 1/vz
    ξz_out = mpz/vpz - mz/vz

    # return message
    return NormalWeightedMeanPrecision(ξz_out, wz_out)

end

@rule Probit(:in, MomentMatching) (q_out::PointMass, m_in::UnivariateNormalDistributionsFamily) = begin

    # extract parameters
    mz, vz = mean_cov(m_in)
    p = mean(q_out)
    @assert p >= zero(p) && p <= one(p) "The Probit node only accepts messages on its output with values between 0 and 1."

    # calculate auxiliary variables
    γ = mz/sqrt(1+vz)

    # calculate unnormalized moments of g
    umom1_g = normcdf(γ)*mz + vz*normpdf(γ)/sqrt(1+vz)
    umom2_g = 2*mz*umom1_g + (vz - mz^2)*normcdf(γ) - vz^2*γ*normpdf(γ)/(1+vz)

    # calculate moments of posterior
    mom0_pz = 1 - p + (2*p - 1)*normcdf(γ)
    mom1_pz = 1/mom0_pz*((1-p)*mz + (2*p-1)*umom1_g)
    mom2_pz = 1/mom0_pz*((1-p)*(vz + mz^2) + (2*p-1)*umom2_g)

    # calculate parameters of posterior
    mpz = mom1_pz
    vpz = mom2_pz - mom1_pz^2
    vpz = min(max(vpz, tiny), vz - tiny) # ensure variance of marginal is not larger than the variance of the cavity distribution.

    # calculate parameters of outgoing message
    wz_out = 1/vpz - 1/vz
    ξz_out = mpz/vpz - mz/vz

    # return message
    return NormalWeightedMeanPrecision(ξz_out, wz_out)

end