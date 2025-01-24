
@rule Transition(:a, Marginalisation) (q_out::Any, q_in::Categorical) = begin
    return MatrixDirichlet(collect(probvec(q_out)) * probvec(q_in)' .+ 1)
end

@rule Transition(:a, Marginalisation) (q_out_in::Contingency, meta::Any) = begin
    return MatrixDirichlet(components(q_out_in) .+ 1)
end

ReactiveMP.rule(
    fform::Type{<:Transition},
    on::Val{:a},
    vconstraint::Marginalisation,
    messages_names::Nothing,
    messages::Nothing,
    marginals_names::Val{m_names} where {m_names},
    marginals::Tuple{<:Marginal{<:Contingency}},
    meta::Any,
    addons::Any,
    ::Any
) = TensorDirichlet(components(getdata(first(marginals))) .+ 1), addons
