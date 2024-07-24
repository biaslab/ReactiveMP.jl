using TupleTools, AdvancedHMC, LogDensityProblems

import Distributions: Distribution
import BayesBase: AbstractContinuousGenericLogPdf

@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{1, Any}, meta::DeltaMeta{M}) where {M <: CVIProjection} = begin
    method = ReactiveMP.getmethod(meta)
    g      = getnodefn(meta, Val(:out))
    m_in   = first(m_ins)
    T  = try 
            ExponentialFamily.exponential_family_typetag(m_in)
        catch e
            first(getcviprojectiontypes(method)[:in])
        end
    prj = ProjectedTo(T, size(m_in)...; parameters = getcviprojectionparameters(method))
    q = project_to(prj, (z) -> logpdf(m_out, g(z)), m_in)
    # q = project_to(prj, (z) -> logpdf(m_out, g(z)) + logpdf(m_in,z)) ## this is still valid if m_in is not in ef

    return FactorizedJoint((q,))
end

#We would need a mixed HMC method to be able to sample from mixed continuous and discrete 
@marginalrule DeltaFn(:ins) (m_out::Any, m_ins::ManyOf{N, Any}, meta::DeltaMeta{M}) where {N, M <: CVIProjection} = begin
    nodefunction = getnodefn(meta, Val(:out))
    marginal = try
        #hmc based method will fail if there are discrete rvs
        cvi_compute_in_marginals(HMCBased(), nodefunction, m_out, m_ins, meta, N)
    catch
        #drop indices based method will fail if m_ins are not sampleable
        cvi_compute_in_marginals(DropIndicesBased(), nodefunction, m_out, m_ins, meta, N)
    end
    return marginal
end

function cvi_compute_in_marginals(based::HMCBased,node_function, m_out, m_ins, meta::DeltaMeta{M}, N) where {M <: CVIProjection} 
    method              = ReactiveMP.getmethod(meta)
    rng                 = getcvirng(method)
    number_marginal_samples  = getcvimarginalsamplesno(based, method)

    var_form_ins, dims_in, sum_dim_in, start_indices, manifolds, natural_parameters_efs, initial_sample = __auxiliary_variables(rng, m_ins, method, N)
    joint_logpdf = (x) -> logpdf(m_out, node_function(ReactiveMP.__splitjoin(x, dims_in)...)) + mapreduce((m_in,k,T) -> log_target_adjusted_log_pdf(T, m_in, getindex(dims_in, k))(ReactiveMP.__splitjoinelement(x, getindex(start_indices, k), getindex(dims_in, k))), +, m_ins, 1:N, var_form_ins)
    log_target_density  = LogTargetDensity(sum_dim_in, joint_logpdf)

    samples            = hmc_samples(rng, sum_dim_in, log_target_density, initial_sample; no_samples = number_marginal_samples + 1)
    samples_collection = map(k -> map((sample) -> ReactiveMP.__splitjoinelement(sample, getindex(start_indices,k), getindex(dims_in, k)), samples), 1:N)
    
    function projection_to_ef(i)
        manifold = getindex(manifolds, i)
        naturalparameters = getindex(natural_parameters_efs,i)
        f = let @views sample = samples_collection[i]
            (M, p) -> begin
                return targetfn(M, p, sample)
            end
        end
        g = let @views sample = samples_collection[i]
            (M, p) -> begin 
                return grad_targetfn(M, p, sample)
            end
        end
        return convert(ExponentialFamilyDistribution, manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, naturalparameters; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
        )
     
    end
    result = FactorizedJoint(map(d -> convert(Distribution, d), ntuple(i -> projection_to_ef(i), N)))
    return result

end

function cvi_compute_in_marginals(based::HMCBased, node_function, m_out::FactorizedJoint, m_ins, meta::DeltaMeta{M}, N) where {M <: CVIProjection} 
    method              = ReactiveMP.getmethod(meta)
    rng                 = getcvirng(method)
    number_marginal_samples  = getcvimarginalsamplesno(based, method)
    var_form_ins, dims_in, sum_dim_in, start_indices, manifolds, natural_parameters_efs, initial_sample = __auxiliary_variables(rng, m_ins, method, N)
    
    components_out   = components(m_out)
    components_length = length(components_out)
    @assert components_length > 1 "Length of the FactorizedJoint $m_out should be greater than 1."
    components_sizes = map(size, components_out)
    prod_dims_components  = map(prod, components_sizes)
    
    cum_lengths_components  = mapreduce(d -> d+1, vcat, cumsum(prod_dims_components))
    start_indices_components = append!([1], cum_lengths_components[1:N-1])

    joint_logpdf_components = (x) -> mapreduce((component,sz,k) -> logpdf(component, ReactiveMP.__splitjoinelement(x, getindex(start_indices_components,k) ,sz)), +,components_out, components_sizes, 1:components_length)

    joint_logpdf = (x) -> joint_logpdf_components(node_function(ReactiveMP.__splitjoin(x, dims_in)...)) + mapreduce((m_in,k,T) -> log_target_adjusted_log_pdf(T, m_in, getindex(dims_in, k))(ReactiveMP.__splitjoinelement(x, getindex(start_indices, k), getindex(dims_in, k))), +, m_ins, 1:N, var_form_ins)
    log_target_density  = LogTargetDensity(sum_dim_in, joint_logpdf)

    samples            = hmc_samples(rng, sum_dim_in, log_target_density, initial_sample; no_samples = number_marginal_samples + 1)
    samples_collection = map(k -> map((sample) -> ReactiveMP.__splitjoinelement(sample, getindex(start_indices,k), getindex(dims_in, k)), samples), 1:N)
    
    function projection_to_ef(i)
        manifold = getindex(manifolds, i)
        naturalparameters = getindex(natural_parameters_efs,i)
        f = let @views sample = samples_collection[i]
            (M, p) -> begin
                return targetfn(M, p, sample)
            end
        end
        g = let @views sample = samples_collection[i]
            (M, p) -> begin 
                return grad_targetfn(M, p, sample)
            end
        end
        return convert(ExponentialFamilyDistribution, manifold,
            ExponentialFamilyProjection.Manopt.gradient_descent(manifold, f, g, naturalparameters; direction = ExponentialFamilyProjection.BoundedNormUpdateRule(1))
        )
     
    end
    result = FactorizedJoint(map(d -> convert(Distribution, d), ntuple(i -> projection_to_ef(i), N)))
    return result

end


function cvi_compute_in_marginals(based::DropIndicesBased, nodefunction, m_out, m_ins, meta::DeltaMeta{M}, N) where {M <: CVIProjection} 
    method = ReactiveMP.getmethod(meta)
    rng = getcvirng(method)
    number_marginal_samples  = getcvimarginalsamplesno(based, method)
    prj  = getcviprojectionparameters(method)

    pre_samples = zip(map(m_in_k -> ReactiveMP.cvilinearize(rand(rng, m_in_k, number_marginal_samples)), m_ins)...)

    logp_nc_drop_index = let g = nodefunction, pre_samples = pre_samples
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> ReactiveMP.TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> g(s...), samples)
            logpdfs = map(out -> logpdf(m_out, out), t_samples)
            return mean(logpdfs)
        end
    end

    optimize_natural_parameters = let m_ins = m_ins, logp_nc_drop_index = logp_nc_drop_index
        (i, pre_samples) -> begin
            # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
            df = let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index
                (z) -> logp_nc_drop_index(z, i, pre_samples)
            end
            logp = convert(promote_variate_type(variate_form(typeof(first(m_ins))), BayesBase.AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)

            T = ExponentialFamily.exponential_family_typetag(m_ins[i])
            prj = ProjectedTo(T, size(m_ins[i])...; parameters = something(ExponentialFamilyProjection.DefaultProjectionParameters()))

            return project_to(prj, logp, m_ins[i])
        end
    end

    return FactorizedJoint(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))
end


function cvi_compute_in_marginals(based::DropIndicesBased,nodefunction,m_out::FactorizedJoint, m_ins, meta::DeltaMeta{M}, N) where {M <: CVIProjection}
    method              = ReactiveMP.getmethod(meta)
    rng                 = getcvirng(method)
    number_marginal_samples  = getcvimarginalsamplesno(based, method)

    components_out   = components(m_out)
    components_length = length(components_out)
    @assert components_length > 1 "Length of the FactorizedJoint $m_out should be greater than 1."
    components_sizes = map(size, components_out)
    prod_dims_components  = map(prod, components_sizes)
    joint_logpdf_components = (x) -> mapreduce((component,sz,k) -> logpdf(component, ReactiveMP.__splitjoinelement(x, getindex(start_indices_components,k) ,sz)), +,components_out, components_sizes, 1:components_length)
    cum_lengths_components  = mapreduce(d -> d+1, vcat, cumsum(prod_dims_components))
    start_indices_components = append!([1], cum_lengths_components[1:N-1])

    pre_samples = zip(map(m_in_k -> ReactiveMP.cvilinearize(rand(rng, m_in_k, number_marginal_samples)), m_ins)...)


    logp_nc_drop_index = let g = nodefunction, pre_samples = pre_samples
        (z, i, pre_samples) -> begin
            samples = map(ttuple -> ReactiveMP.TupleTools.insertat(ttuple, i, (z,)), pre_samples)
            t_samples = map(s -> g(s...), samples)
            logpdfs = map(out -> joint_logpdf_components(out), t_samples)
            return mean(logpdfs)
        end
    end

    optimize_natural_parameters = let m_ins = m_ins, logp_nc_drop_index = logp_nc_drop_index
        (i, pre_samples) -> begin
            # Create an `AbstractContinuousGenericLogPdf` with an unspecified domain and the transformed `logpdf` function
            df = let i = i, pre_samples = pre_samples, logp_nc_drop_index = logp_nc_drop_index
                (z) -> logp_nc_drop_index(z, i, pre_samples)
            end
            logp = convert(promote_variate_type(variate_form(typeof(first(m_ins))), BayesBase.AbstractContinuousGenericLogPdf), UnspecifiedDomain(), df)

            T = ExponentialFamily.exponential_family_typetag(m_ins[i])
            prj = ProjectedTo(T, size(m_ins[i])...; parameters = something(ExponentialFamilyProjection.DefaultProjectionParameters()))

            return project_to(prj, logp, m_ins[i])
        end
    end

    return FactorizedJoint(ntuple(i -> optimize_natural_parameters(i, pre_samples), length(m_ins)))

end



