
@rule DeltaFn(:out, Marginalisation) (m_ins::ManyOf{N, NormalDistributionsFamily}, meta::DeltaMeta{M}) where {N, M <: Unscented} = begin
    return approximate(getmethod(meta), getnodefn(Val(:out)), m_ins)
end
