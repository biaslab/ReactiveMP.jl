using TupleTools

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf

function create_project_to_ins(::CVIProjection, ::Nothing, m_in::Any)
    T = ExponentialFamily.exponential_family_typetag(m_in)
    ef_in = convert(ExponentialFamilyDistribution, m_in)
    conditioner = getconditioner(ef_in)
    return ProjectedTo(T, size(m_in)...; conditioner = conditioner, parameters = ExponentialFamilyProjection.DefaultProjectionParameters())
end

function create_project_to_ins(::CVIProjection, form::ProjectedTo, ::Any)
    return form
end

function create_project_to_ins(::CVIProjection, params::ProjectionParameters, m_in::Any)
    T = ExponentialFamily.exponential_family_typetag(m_in)
    ef_in = convert(ExponentialFamilyDistribution, m_in)
    conditioner = getconditioner(ef_in)
    return ProjectedTo(T, size(m_in)...; conditioner = conditioner, parameters = params)
end

function create_project_to_ins(method::CVIProjection, m_in::Any, k::Int)
    form = ReactiveMP.get_kth_in_form(method, k)
    return create_project_to_ins(method, form, m_in)
end

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    g = getnodefn(meta, Val(:out))

    m_in = first(m_ins)
    ef_in = convert(ExponentialFamilyDistribution, m_in)
    # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
    F = promote_variate_type(variate_form(typeof(m_in)), BayesBase.AbstractContinuousGenericLogPdf)
    f = convert(F, UnspecifiedDomain(), (z) -> logpdf(m_out, g(z)))

    prj = create_project_to_ins(method, m_in, 1)
    q = project_to(prj, f, first(m_ins))

    return FactorizedJoint((q,))
end

function create_density_function(forms_match, i, pre_samples, logp_nc_drop_index, m_in)
    if forms_match
        return z -> logp_nc_drop_index(z, i, pre_samples)
    end
    return z -> logp_nc_drop_index(z, i, pre_samples) + logpdf(m_in, z)
end

function optimize_parameters(i, pre_samples, m_ins, logp_nc_drop_index, method)
    m_in = m_ins[i]
    default_type = ExponentialFamily.exponential_family_typetag(m_in)
    prj = create_project_to_ins(method, m_in, i)

    typeform = ExponentialFamilyProjection.get_projected_to_type(prj)
    dims = ExponentialFamilyProjection.get_projected_to_dims(prj)
    forms_match = typeform === default_type && dims == size(m_in)

    df = create_density_function(forms_match, i, pre_samples, logp_nc_drop_index, m_in)
    logp = convert(promote_variate_type(variate_form(typeof(m_in)), BayesBase.AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)

    return forms_match ? project_to(prj, logp, m_in) : project_to(prj, logp)
end

function generate_samples(rng, ::Nothing, m_ins, sampling_strategy::FullSampling)
    return zip(map(m_in -> ReactiveMP.cvilinearize(rand(rng, m_in, sampling_strategy.samples)), m_ins)...)
end

function generate_samples(::Any, ::Nothing, m_ins, ::MeanBased)
    return zip(map(m_in -> [mean(m_in)], m_ins)...)
end

function generate_samples(rng, proposal_distribution::FactorizedJoint, ::Any, sampling_strategy::FullSampling)
    return zip(map(q_in -> ReactiveMP.cvilinearize(rand(rng, q_in, sampling_strategy.samples)), proposal_distribution.multipliers)...)
end

function generate_samples(::Any, proposal_distribution::FactorizedJoint, ::Any, ::MeanBased)
    return zip(map(q_in -> [mean(q_in)], proposal_distribution.multipliers)...)
end

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    rng = method.rng
    proposal_distribution_container = method.proposal_distribution
    sampling_strategy = method.sampling_strategy

    pre_samples = generate_samples(rng, proposal_distribution_container.distribution, m_ins, sampling_strategy)

    logp_nc_drop_index = let g = getnodefn(meta, Val(:out)), pre_samples = pre_samples
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> ReactiveMP.TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> g(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return mean(logpdfs)
        end
    end

    optimize_natural_parameters = let m_ins = m_ins, logp_nc_drop_index = logp_nc_drop_index
        (i, pre_samples) -> begin
            m_in = m_ins[i]
            default_type = ExponentialFamily.exponential_family_typetag(m_in)

            prj = create_project_to_ins(method, m_in, i)

            typeform = ExponentialFamilyProjection.get_projected_to_type(prj)
            dims = ExponentialFamilyProjection.get_projected_to_dims(prj)
            forms_match = typeform === default_type && dims == size(m_in)

            # Create log probability function
            df = if forms_match
                let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index
                    (z) -> logp_nc_drop_index(z, i, pre_samples)
                end
            else
                let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index, m_in = m_in
                    (z) -> logp_nc_drop_index(z, i, pre_samples) + logpdf(m_in, z)
                end
            end

            logp = convert(promote_variate_type(variate_form(typeof(m_in)), BayesBase.AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)

            return forms_match ? project_to(prj, logp, m_in) : project_to(prj, logp)
        end
    end

    result = FactorizedJoint(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))
    proposal_distribution_container.distribution = result
    return result
end
