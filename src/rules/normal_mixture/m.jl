export rule

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::Union{Categorical, PointMass{<:Vector}}, q_p::Any) = begin
    pv    = probvec(q_switch)
    T     = eltype(pv)
    z_bar = clamp.(pv, tiny, one(T) - tiny)

    F = variate_form(typeof(q_out))

    return convert(promote_variate_type(F, NormalMeanPrecision), mean(q_out), z_bar[k] * mean(q_p))
end

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::PointMass{<:Real}, q_p::Any) = begin
    error(
        "Cannot handle switch with Integer values. The switch variable should be a one-hot encoded vector where each element represents the probability of being in that state. Please convert your integer switch values to one-hot encoded vectors before using this rule."
    )
end
