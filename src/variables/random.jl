export RandomVariable, RandomVariableCreationOptions, randomvar

import Base: show

## Random variable implementation

mutable struct RandomVariable <: AbstractVariable
    name                :: Symbol
    collection_type     :: AbstractVariableCollectionType
    input_messages      :: Vector{MessageObservable{AbstractMessage}} 
    output_messages     :: Vector{MessageObservable{Message}}
    output_initialised  :: Bool
    output_cache        :: Union{Nothing, EqualityChain}
    marginal            :: MarginalObservable
    pipeline            :: AbstractPipelineStage
    proxy_variables     
    prod_constraint     
    prod_strategy       
    marginal_form_constraint     
    marginal_form_check_strategy
    messages_form_constraint
    messages_form_check_strategy
end

Base.show(io::IO, randomvar::RandomVariable) = print(io, "RandomVariable(", indexed_name(randomvar), ")")

struct RandomVariableCreationOptions{L, P, C, S, T, G, M, N}
    pipeline                     :: L
    proxy_variables              :: P
    prod_constraint              :: C
    prod_strategy                :: S
    marginal_form_constraint     :: T
    marginal_form_check_strategy :: G
    messages_form_constraint     :: M
    messages_form_check_strategy :: N
end

function RandomVariableCreationOptions()
    # Return default settings
    return RandomVariableCreationOptions(
        EmptyPipelineStage(),             # pipeline
        nothing,                          # proxy_variables
        ProdAnalytical(),                 # prod_constraint
        FoldLeftProdStrategy(),           # prod_strategy
        UnspecifiedFormConstraint(),      # marginal_form_constraint
        FormConstraintCheckPickDefault(), # marginal_form_check_strategy
        UnspecifiedFormConstraint(),      # messages_form_constraint
        FormConstraintCheckPickDefault()  # messages_form_check_strategy
    )
end

randomvar(name::Symbol)                                                  = randomvar(RandomVariableCreationOptions(), name, VariableIndividual())
randomvar(name::Symbol, collection_type::AbstractVariableCollectionType) = randomvar(RandomVariableCreationOptions(), name, collection_type)
randomvar(name::Symbol, length::Int)                                     = randomvar(RandomVariableCreationOptions(), name, length)
randomvar(name::Symbol, dims::Tuple)                                     = randomvar(RandomVariableCreationOptions(), name, dims)
randomvar(name::Symbol, dims::Vararg{Int})                               = randomvar(RandomVariableCreationOptions(), name, dims)

randomvar(options::RandomVariableCreationOptions, name::Symbol)                    = randomvar(options, name, VariableIndividual())
randomvar(options::RandomVariableCreationOptions, name::Symbol, dims::Vararg{Int}) = randomvar(options, name, dims)

function randomvar(options::RandomVariableCreationOptions, name::Symbol, collection_type::AbstractVariableCollectionType) 
    return RandomVariable(
        name, 
        collection_type, 
        Vector{MessageObservable{AbstractMessage}}(), 
        Vector{MessageObservable{Message}}(), 
        false, 
        nothing, 
        MarginalObservable(), 
        options.pipeline, 
        options.proxy_variables, 
        options.prod_constraint, 
        options.prod_strategy, 
        options.marginal_form_constraint, 
        options.marginal_form_check_strategy,
        options.messages_form_constraint, 
        options.messages_form_check_strategy
    )
end

function randomvar(options::RandomVariableCreationOptions, name::Symbol, length::Int)
    return map(i -> randomvar(options, name, VariableVector(i)), 1:length)
end

function randomvar(options::RandomVariableCreationOptions, name::Symbol, dims::Tuple)
    indices = CartesianIndices(dims)
    size = axes(indices)
    return map(i -> randomvar(options, name, VariableArray(size, i)), indices)
end

degree(randomvar::RandomVariable)              = length(randomvar.input_messages)
name(randomvar::RandomVariable)                = randomvar.name
proxy_variables(randomvar::RandomVariable)     = randomvar.proxy_variables
collection_type(randomvar::RandomVariable)     = randomvar.collection_type
equality_chain(randomvar::RandomVariable)      = randomvar.equality_chain
prod_constraint(randomvar::RandomVariable)     = randomvar.prod_constraint
prod_strategy(randomvar::RandomVariable)       = randomvar.prod_strategy

marginal_form_constraint(randomvar::RandomVariable)     = randomvar.marginal_form_constraint
marginal_form_check_strategy(randomvar::RandomVariable) = _marginal_form_check_strategy(randomvar.marginal_form_check_strategy, randomvar)

_marginal_form_check_strategy(::FormConstraintCheckPickDefault, randomvar::RandomVariable) = default_form_check_strategy(marginal_form_constraint(randomvar))
_marginal_form_check_strategy(form_check_strategy, randomvar::RandomVariable)              = form_check_strategy

messages_form_constraint(randomvar::RandomVariable)     = randomvar.messages_form_constraint
messages_form_check_strategy(randomvar::RandomVariable) = _messages_form_check_strategy(randomvar.messages_form_check_strategy, randomvar)

_messages_form_check_strategy(::FormConstraintCheckPickDefault, randomvar::RandomVariable) = default_form_check_strategy(messages_form_constraint(randomvar))
_messages_form_check_strategy(form_check_strategy, randomvar::RandomVariable)              = form_check_strategy

isproxy(randomvar::RandomVariable) = proxy_variables(randomvar) !== nothing

israndom(::RandomVariable)                     = true
israndom(::AbstractArray{ <: RandomVariable }) = true

messages_prod_fn(randomvar::RandomVariable) = messages_prod_fn(prod_strategy(randomvar), prod_constraint(randomvar), messages_form_constraint(randomvar), messages_form_check_strategy(randomvar))
marginal_prod_fn(randomvar::RandomVariable) = marginal_prod_fn(prod_strategy(randomvar), prod_constraint(randomvar), marginal_form_constraint(randomvar), marginal_form_check_strategy(randomvar))

getlastindex(randomvar::RandomVariable) = degree(randomvar) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.input_messages[index]

function messageout(randomvar::RandomVariable, index::Int) 
    if randomvar.output_initialised === false
        initialize_output_messages!(randomvar)
    end
    return @inbounds randomvar.output_messages[index]
end

get_pipeline_stages(randomvar::RandomVariable)        = randomvar.pipeline
add_pipeline_stage!(randomvar::RandomVariable, stage) = randomvar.pipeline = (randomvar.pipeline + stage)

_getmarginal(randomvar::RandomVariable)              = randomvar.marginal
_setmarginal!(randomvar::RandomVariable, observable) = connect!(_getmarginal(randomvar), observable)
_makemarginal(randomvar::RandomVariable)             = collectLatest(AbstractMessage, Marginal, randomvar.input_messages, marginal_prod_fn(randomvar))

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === degree(randomvar) + 1
        push!(randomvar.input_messages, messagein)
    else
        error("Inconsistent state in setmessagein! function for random variable $(randomvar). `index` should be equal to `degree(randomvar) + 1 = $(degree(randomvar) + 1)`, $(index) is given instead")
    end
end

function activate!(model, randomvar::RandomVariable)
    # `5` here is empirical observation, maybe we can come up with better heuristic?
    # in case if number of connections is large we use cache equality nodes chain structure 
    if degree(randomvar) > 5
        chain_pipeline  = schedule_on(global_reactive_scheduler(getoptions(model)))
        chain_prod_fn   = messages_prod_fn(randomvar)
        randomvar.output_cache = EqualityChain(randomvar.input_messages, chain_pipeline, chain_prod_fn)
    end

    _setmarginal!(randomvar, _makemarginal(randomvar))

    return nothing
end

initialize_output_messages!(randomvar::RandomVariable) = initialize_output_messages!(randomvar.output_cache, randomvar)

# Generic fallback for variables with small number of connected nodes, somewhere <= 5
# We do not create equality chain in this cases, but simply do eager product
function initialize_output_messages!(::Nothing, randomvar::RandomVariable)
    d          = degree(randomvar)
    outputmsgs = randomvar.output_messages
    inputmsgs  = randomvar.input_messages
    prod_fn    = messages_prod_fn(randomvar)

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
        outputmsg     = collectLatest(AbstractMessage, Message, skipindex(inputmsgs, i), prod_fn)
        connect!(outputmsgs[i], outputmsg)
    end

    randomvar.output_initialised = true

    return nothing
end

# Equality chain initialisation for variables with large number of connected nodes, somewhere > 5
# In this cases it is more efficient to create an equality chain structure, but it does allocate way more memory
function initialize_output_messages!(chain::EqualityChain, randomvar::RandomVariable)
    d          = degree(randomvar)
    outputmsgs = randomvar.output_messages

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
    end

    initialize!(chain, outputmsgs)
    randomvar.output_initialised = true

    return nothing
end