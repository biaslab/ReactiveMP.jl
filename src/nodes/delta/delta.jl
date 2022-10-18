export DeltaMeta

"""
    DeltaMeta(method = ..., [ inverse = ... ])

`DeltaMeta` structure specifies the approximation method for the outbound messages in the `DeltaFn` node. 

# Arguments
- `method`: required, the approximation method, currently supported methods are [`Linearization`](@ref), [`Unscented`](@ref) and [`CVI`](@ref).
- `inverse`: optional, some methods benefit from the explicit definition of the inverse function.

Is is also possible to pass the `AbstractApproximationMethod` to the meta of the delta node directly. In this case `inverse` is set to `nothing`.

See also: [`DeltaFn`](@ref), [`Linearization`](@ref), [`Unscented`](@ref), [`CVI`](@ref).
"""
struct DeltaMeta{M, I}
    method  :: M
    inverse :: I
end

function DeltaMeta(; method::M, inverse::I = nothing) where { M, I }
    return DeltaMeta{M, I}(method, inverse)
end

getmethod(meta::DeltaMeta)          = meta.method
getinverse(meta::DeltaMeta)         = meta.inverse
getinverse(meta::DeltaMeta, k::Int) = meta.inverse[k]

struct DeltaFnCallableWrapper{F} end

(::Type{DeltaFnCallableWrapper{F}})(args...) where {F} = F.instance(args...)

struct DeltaFn{F} end
struct DeltaFnNode{F, N, L, M} <: AbstractFactorNode
    fn::F

    out::NodeInterface
    ins::NTuple{N, IndexedNodeInterface}

    localmarginals :: L
    metadata       :: M
end

as_node_symbol(::Type{DeltaFn{ReactiveMP.DeltaFnCallableWrapper{F}}}) where {F} = Symbol(:DeltaFn, string(F))

functionalform(factornode::DeltaFnNode{F}) where {F}      = DeltaFn{DeltaFnCallableWrapper{F}}
sdtype(factornode::DeltaFnNode)                           = Deterministic()
interfaces(factornode::DeltaFnNode)                       = (factornode.out, factornode.ins...)
factorisation(factornode::DeltaFnNode{F, N}) where {F, N} = ntuple(identity, N + 1)
localmarginals(factornode::DeltaFnNode)                   = factornode.localmarginals.marginals
localmarginalnames(factornode::DeltaFnNode)               = map(name, localmarginals(factornode))
metadata(factornode::DeltaFnNode)                         = factornode.metadata

collect_meta(::Type{DeltaFn}, something)       = error("Delta node requires meta specification with the `where { meta = ... }` in the `@model` macro or with the separate `@meta` specification. See documentation for the `DeltaMeta`.")
collect_meta(::Type{DeltaFn}, meta::DeltaMeta) = meta
collect_meta(::Type{DeltaFn}, method::AbstractApproximationMethod) = DeltaMeta(method = method, inverse = nothing)

# For missing rules error msg
rule_method_error_extract_fform(f::Type{<:DeltaFn}) = "DeltaFn{f}"

function interfaceindex(factornode::DeltaFnNode, iname::Symbol)
    # Julia's constant propagation should compile-out this if-else branch
    if iname === :out
        return 1
    elseif iname === :ins || iname === :in
        return 2
    else
        error("Unknown interface ':$(iname)' for nonlinear delta fn [ $(functionalform(factornode)) ] node")
    end
end

function __make_delta_fn_node(fn::F, options::FactorNodeCreationOptions, out::AbstractVariable, ins::NTuple{N, <:AbstractVariable}) where {F <: Function, N}
    out_interface = NodeInterface(:out, Marginalisation())
    ins_interface = ntuple(i -> IndexedNodeInterface(i, NodeInterface(:in, Marginalisation())), N)

    out_index = getlastindex(out)
    connectvariable!(out_interface, out, out_index)
    setmessagein!(out, out_index, messageout(out_interface))

    foreach(zip(ins_interface, ins)) do (in_interface, in_var)
        in_index = getlastindex(in_var)
        connectvariable!(in_interface, in_var, in_index)
        setmessagein!(in_var, in_index, messageout(in_interface))
    end

    localmarginals = FactorNodeLocalMarginals((FactorNodeLocalMarginal(1, 1, :out), FactorNodeLocalMarginal(2, 2, :ins)))
    meta           = collect_meta(DeltaFn{F}, metadata(options))
    pipeline       = getpipeline(options)

    if !isnothing(pipeline)
        @warn "Delta node does not support the `pipeline` option."
    end

    return DeltaFnNode(fn, out_interface, ins_interface, localmarginals, meta)
end

function make_node(fform::F, options::FactorNodeCreationOptions, autovar::AutoVar, args::Vararg{<:AbstractVariable}) where {F <: Function}
    out = randomvar(getname(autovar))
    return __make_delta_fn_node(fform, options, out, args), out
end

function make_node(fform::F, options::FactorNodeCreationOptions, args::Vararg{<:AbstractVariable}) where {F <: Function}
    return __make_delta_fn_node(fform, options, args[1], args[2:end])
end

# By default all `meta` objects fallback to the `DeltaFnDefaultRuleLayout`
# We, however, allow for specific approximation methods to override the default `DeltaFn` rule layout for better efficiency
deltafn_rule_layout(factornode::DeltaFnNode)                  = deltafn_rule_layout(factornode, metadata(factornode))
deltafn_rule_layout(factornode::DeltaFnNode, meta::DeltaMeta) = deltafn_rule_layout(factornode, getmethod(meta), getinverse(meta))

include("layouts/default.jl")

deltafn_rule_layout(::DeltaFnNode, ::AbstractApproximationMethod, inverse::Nothing)                       = DeltaFnDefaultRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::AbstractApproximationMethod, inverse::Function)                      = DeltaFnDefaultKnownInverseRuleLayout()
deltafn_rule_layout(::DeltaFnNode, ::AbstractApproximationMethod, inverse::NTuple{N, Function}) where {N} = DeltaFnDefaultKnownInverseRuleLayout()

function activate!(model, factornode::DeltaFnNode)
    # `DeltaFn` node may change rule arguments layout depending on the `meta`
    # This feature is similar to `functional_dependencies` for a regular `FactorNode` implementation
    return activate!(model, factornode, deltafn_rule_layout(factornode))
end

# DeltaFn is very special, so it has a very special `activate!` function
function activate!(model, factornode::DeltaFnNode, layout)
    foreach(interfaces(factornode)) do interface
        (connectedvar(interface) !== nothing) || error("Empty variable on interface $(interface) of node $(factornode)")
    end

    # First we declare local marginal for `out` edge
    deltafn_apply_layout(layout, Val(:q_out), model, factornode)

    # Second we declare how to compute a joint marginal over all inbound edges
    deltafn_apply_layout(layout, Val(:q_ins), model, factornode)

    # Second we declare message passing logic for out interface
    deltafn_apply_layout(layout, Val(:m_out), model, factornode)

    # At last we declare message passing logic for input interfaces
    deltafn_apply_layout(layout, Val(:m_in), model, factornode)
end

# DeltaFn has a bit a non-standard interface layout so it has a specialised `score` function too

function score(::Type{T}, objective::BetheFreeEnergy, ::FactorBoundFreeEnergy, ::Deterministic, node::DeltaFnNode, scheduler) where {T <: InfCountingReal}

    # TODO (make a function for `node.localmarginals.marginals[2]`)
    qinsmarginal = apply_skip_filter(getstream(node.localmarginals.marginals[2]), marginal_skip_strategy(objective))

    stream  = qinsmarginal |> schedule_on(scheduler)
    mapping = (marginal) -> convert(T, -score(DifferentialEntropy(), marginal))

    return apply_diagnostic_check(objective, node, stream |> map(T, mapping))
end
