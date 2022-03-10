export ModelOptions, model_options
export Model
export AutoVar
export getoptions, getconstraints, getmeta
export getnodes, getrandom, getconstant, getdata
export activate!, repeat!
export UntilConvergence
# export MarginalsEagerUpdate, MarginalsPureUpdate

import Base: show, getindex, haskey, firstindex, lastindex

# Marginals update strategies 

# struct MarginalsEagerUpdate end
# struct MarginalsPureUpdate end

# Model Options

struct ModelOptions{P, F, S}
    pipeline                   :: P
    default_factorisation      :: F
    global_reactive_scheduler  :: S
end

model_options() = model_options(NamedTuple{()}(()))

available_option_names(::Type{ <: ModelOptions }) = (
    :pipeline, 
    :default_factorisation,
    :global_reactive_scheduler, 
    :limit_stack_depth
)

function model_options(options::NamedTuple)
    pipeline                  = EmptyPipelineStage()
    default_factorisation     = UnspecifiedConstraints()
    global_reactive_scheduler = AsapScheduler()

    if haskey(options, :pipeline)
        pipeline = options[:pipeline]
    end

    if haskey(options, :default_factorisation)
        default_factorisation = options[:default_factorisation]
    end

    if haskey(options, :global_reactive_scheduler)
        global_reactive_scheduler = options[:global_reactive_scheduler]
    elseif haskey(options, :limit_stack_depth)
        global_reactive_scheduler = LimitStackScheduler(options[:limit_stack_depth]...)
    end

    for key::Symbol in setdiff(union(available_option_names(ModelOptions), fields(options)), available_option_names(ModelOptions))
        @warn "Unknown option key: $key = $(options[key])"
    end

    return ModelOptions(
        pipeline,
        default_factorisation,
        global_reactive_scheduler
    )
end

global_reactive_scheduler(options::ModelOptions) = options.global_reactive_scheduler
get_pipeline_stages(options::ModelOptions)       = options.pipeline
default_factorisation(options::ModelOptions)     = options.default_factorisation

# Model

struct Model{C, M, O}
    constraints :: C
    meta        :: M
    options     :: O
    nodes       :: Vector{AbstractFactorNode}
    random      :: Vector{RandomVariable}
    constant    :: Vector{ConstVariable}
    data        :: Vector{DataVariable}
    vardict     :: Dict{Symbol, Any}
end

Base.show(io::IO, ::Type{ <: Model }) = print(io, "Model")
Base.show(io::IO, model::Model)       = print(io, "Model()")

Model() = Model(DefaultConstraints, DefaultMeta, model_options())

Model(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }) = Model(constraints, DefaultMeta, model_options())
Model(meta::Union{ UnspecifiedMeta, MetaSpecification })                      = Model(DefaultConstraints, meta, model_options())
Model(options::NamedTuple)                                                    = Model(DefaultConstraints, DefaultMeta, model_options(options))

Model(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }, options::NamedTuple) = Model(constraints, DefaultMeta, model_options(options))
Model(meta::Union{ UnspecifiedMeta, MetaSpecification }, options::NamedTuple)                      = Model(DefaultConstraints, meta, model_options(options))

Model(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }, meta::Union{ UnspecifiedMeta, MetaSpecification })                       = Model(constraints, meta, model_options())
Model(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }, meta::Union{ UnspecifiedMeta, MetaSpecification }, options::NamedTuple)  = Model(constraints, meta, model_options(options))

function Model(constraints::C, meta::M, options::O) where { C <: Union{ UnspecifiedConstraints, ConstraintsSpecification }, M <: Union{ UnspecifiedMeta, MetaSpecification }, O <: ModelOptions } 
    return Model{C, M, O}(constraints, meta, options, Vector{FactorNode}(), Vector{RandomVariable}(), Vector{ConstVariable}(), Vector{DataVariable}(), Dict{Symbol, Any}())
end

getconstraints(model::Model) = model.constraints
getmeta(model::Model)        = model.meta
getoptions(model::Model)     = model.options
getnodes(model::Model)       = model.nodes
getrandom(model::Model)      = model.random
getconstant(model::Model)    = model.constant
getdata(model::Model)        = model.data
getvardict(model::Model)     = model.vardict

function Base.getindex(model::Model, symbol::Symbol) 
    vardict = getvardict(model)
    if !haskey(vardict, symbol)
        error("Model has no variable/variables named $(symbol).")
    end
    return getindex(getvardict(model), symbol)
end

function Base.haskey(model::Model, symbol::Symbol)
    return haskey(getvardict(model), symbol)
end

firstindex(model::Model, symbol::Symbol) = firstindex(model, getindex(model, symbol))
lastindex(model::Model, symbol::Symbol)  = lastindex(model, getindex(model, symbol))

firstindex(::Model, ::AbstractVariable) = typemin(Int64)
lastindex(::Model, ::AbstractVariable)  = typemax(Int64)

firstindex(::Model, variables::AbstractVector{ <: AbstractVariable }) = firstindex(variables)
lastindex(::Model, variables::AbstractVector{ <: AbstractVariable })  = lastindex(variables)

add!(vardict::Dict, name::Symbol, entity) = vardict[name] = entity

add!(model::Model, node::AbstractFactorNode)  = begin push!(model.nodes, node); return node end
add!(model::Model, randomvar::RandomVariable) = begin push!(model.random, randomvar); add!(getvardict(model), name(randomvar), randomvar); return randomvar end
add!(model::Model, constvar::ConstVariable)   = begin push!(model.constant, constvar); add!(getvardict(model), name(constvar), constvar); return constvar end
add!(model::Model, datavar::DataVariable)     = begin push!(model.data, datavar); add!(getvardict(model), name(datavar), datavar); return datavar end
add!(model::Model, ::Nothing)                 = nothing
add!(model::Model, collection::Tuple)         = begin foreach((d) -> add!(model, d), collection); return collection end
add!(model::Model, array::AbstractArray)      = begin foreach((d) -> add!(model, d), array); return array end
add!(model::Model, array::AbstractArray{ <: RandomVariable }) = begin append!(model.random, array); add!(getvardict(model), name(first(array)), array); return array end
add!(model::Model, array::AbstractArray{ <: ConstVariable })  = begin append!(model.constant, array); add!(getvardict(model), name(first(array)), array); return array end
add!(model::Model, array::AbstractArray{ <: DataVariable })   = begin append!(model.data, array); add!(getvardict(model), name(first(array)), array); return array end

function activate!(model::Model) 
    filter!(getrandom(model)) do randomvar
        @assert degree(randomvar) !== 1 "Half-edge has been found: $(name(randomvar)). To terminate half-edges 'Uninformative' node can be used."
        return degree(randomvar) >= 2
    end

    foreach(getdata(model)) do datavar
        if !isconnected(datavar)
            @warn "Unused data variable has been found: '$(name(datavar))'. Ignore if '$(name(datavar))' has been used in deterministic nonlinear tranformation."
        end
    end

    filter!(c -> isconnected(c), getconstant(model))
    foreach(r -> activate!(model, r), getrandom(model))
    foreach(n -> activate!(model, n), getnodes(model))
end

# Utility functions

## node

function node_resolve_options(model::Model, options::FactorNodeCreationOptions, fform, variables) 
    return FactorNodeCreationOptions(
        node_resolve_factorisation(model, options, fform, variables),
        node_resolve_meta(model, options, fform, variables),
        getpipeline(options)
    )
end

## constraints 

node_resolve_factorisation(model::Model, options::FactorNodeCreationOptions, fform, variables)            = node_resolve_factorisation(model, options, factorisation(options), fform, variables)
node_resolve_factorisation(model::Model, options::FactorNodeCreationOptions, something, fform, variables) = something
node_resolve_factorisation(model::Model, options::FactorNodeCreationOptions, ::Nothing, fform, variables) = node_resolve_factorisation(model, getconstraints(model), default_factorisation(getoptions(model)), fform, variables)

node_resolve_factorisation(model::Model, constraints, default, fform, variables)                               = error("Cannot resolve factorisation constrains. Both `constraints` and `default_factorisation` option have been set, which is disallowed.")
node_resolve_factorisation(model::Model, ::UnspecifiedConstraints, default, fform, variables)                  = default
node_resolve_factorisation(model::Model, constraints, ::UnspecifiedConstraints, fform, variables)              = resolve_factorisation(constraints, model, fform, variables)
node_resolve_factorisation(model::Model, ::UnspecifiedConstraints, ::UnspecifiedConstraints, fform, variables) = resolve_factorisation(UnspecifiedConstraints(), model, fform, variables)

## meta 

node_resolve_meta(model::Model, options::FactorNodeCreationOptions, fform, variables)            = node_resolve_meta(model, options, metadata(options), fform, variables)
node_resolve_meta(model::Model, options::FactorNodeCreationOptions, something, fform, variables) = something
node_resolve_meta(model::Model, options::FactorNodeCreationOptions, ::Nothing, fform, variables) = resolve_meta(getmeta(model), model, fform, variables)

## randomvar

function randomvar_resolve_options(model::Model, options::RandomVariableCreationOptions, name) 
    qform, qprod = randomvar_resolve_marginal_form_prod(model, options, name)
    mform, mprod = randomvar_resolve_messages_form_prod(model, options, name)

    rprod = resolve_prod_constraint(qprod, mprod)

    qoptions = randomvar_options_set_marginal_form_constraint(options, qform)
    moptions = randomvar_options_set_messages_form_constraint(qoptions, mform)
    roptions = randomvar_options_set_prod_constraint(moptions, rprod)

    return roptions
end

## constraints

randomvar_resolve_marginal_form_prod(model::Model, options::RandomVariableCreationOptions, name)            = randomvar_resolve_marginal_form_prod(model, options, marginal_form_constraint(options), name)
randomvar_resolve_marginal_form_prod(model::Model, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_marginal_form_prod(model::Model, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_marginal_form_prod(model, getconstraints(model), name)

randomvar_resolve_marginal_form_prod(model::Model, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_marginal_form_prod(model::Model, constraints, name)              = resolve_marginal_form_prod(constraints, model, name)

randomvar_resolve_messages_form_prod(model::Model, options::RandomVariableCreationOptions, name)            = randomvar_resolve_messages_form_prod(model, options, messages_form_constraint(options), name)
randomvar_resolve_messages_form_prod(model::Model, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_messages_form_prod(model::Model, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_messages_form_prod(model, getconstraints(model), name)

randomvar_resolve_messages_form_prod(model::Model, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_messages_form_prod(model::Model, constraints, name)              = resolve_messages_form_prod(constraints, model, name)

## variable creation

function randomvar(model::Model, name::Symbol, args...)
    return randomvar(model, RandomVariableCreationOptions(), name, args...)
end

function randomvar(model::Model, options::RandomVariableCreationOptions, name::Symbol, args...) 
    return add!(model, randomvar(randomvar_resolve_options(model, options, name), name, args...))
end

constvar(model::Model, args...)  = add!(model, constvar(args...))
datavar(model::Model, args...)   = add!(model, datavar(args...))

as_variable(model::Model, x)                   = add!(model, as_variable(x))
as_variable(model::Model, v::AbstractVariable) = v
as_variable(model::Model, t::Tuple)            = map((d) -> as_variable(model, d), t)

function make_node(model::Model, options::FactorNodeCreationOptions, fform, args...)
    return add!(model, make_node(fform, node_resolve_options(model, options, fform, args), args...))
end

## AutoVar 

struct AutoVar
    name :: Symbol
end

getname(autovar::AutoVar) = autovar.name

function ReactiveMP.make_node(model::Model, options::FactorNodeCreationOptions, fform, autovar::AutoVar, args::Vararg{ <: ReactiveMP.AbstractVariable })
    proxy     = isdeterministic(sdtype(fform)) ? args : nothing
    rvoptions = ReactiveMP.randomvar_options_set_proxy_variables(proxy)
    var       = add!(model, ReactiveMP.randomvar(rvoptions, ReactiveMP.getname(autovar)))
    node      = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
    return node, var
end

__fform_const_apply(::Type{T}, args...) where T = T(args...)
__fform_const_apply(f::F, args...) where { F <: Function } = f(args...)

function ReactiveMP.make_node(model::Model, options::FactorNodeCreationOptions, fform, autovar::AutoVar, args::Vararg{ <: ReactiveMP.ConstVariable })
    if isstochastic(sdtype(fform))
        var  = add!(model, ReactiveMP.randomvar(ReactiveMP.getname(autovar)))
        node = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
        return node, var
    else
        var  = add!(model, ReactiveMP.constvar(ReactiveMP.getname(autovar), __fform_const_apply(fform, map((d) -> ReactiveMP.getconst(d), args)...)))
        return nothing, var
    end
end

##

# Repeat variational message passing iterations [ EXPERIMENTAL ]

repeat!(model::Model, criterion) = repeat!((_) -> nothing, model, criterion)

function repeat!(callback::Function, model::Model, count::Int)
    for _ in 1:count
        foreach(getdata(model)) do datavar
            resend!(datavar)
        end
        callback(model)
    end
end

struct UntilConvergence{S, F}
    score_type :: S
    tolerance  :: F
    maxcount   :: Int
    mincount   :: Int
end

UntilConvergence(score::S; tolerance::F = 1e-6, maxcount::Int = 10_000, mincount::Int = 0) where { S, F <: Real } = UntilConvergence{S, F}(score, tolerance, maxcount, mincount)
UntilConvergence(; kwargs...)                                                                                     = UntilConvergence(BetheFreeEnergy(); kwargs...)

function repeat!(callback::Function, model::Model, criterion::UntilConvergence{S, F}) where { S, F }

    stopping_fn = let tolerance = criterion.tolerance
        (p) -> abs(p[1] - p[2]) < tolerance
    end

    source = score(F, criterion.score_type, model, AsapScheduler()) |> pairwise() |> map(Bool, stopping_fn)

    is_satisfied     = false
    iterations_count = 0

    subscription = subscribe!(source, lambda(
        on_next     = (v) -> is_satisfied = v,
        on_complete = (v) -> is_satisfied = true
    ))

    while !is_satisfied
        foreach(getdata(model)) do datavar
            resend!(datavar)
        end
        callback(model)
        iterations_count += 1

        if iterations_count <= criterion.mincount
            is_satisfied = false
        end

        if iterations_count >= criterion.maxcount
            is_satisfied = true
        end
    end

    unsubscribe!(subscription)

    return iterations_count
end