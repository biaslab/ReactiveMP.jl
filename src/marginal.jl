export Marginal, getdata, is_clamped, is_initial, as_marginal
export SkipClamped, SkipInitial, SkipClampedAndInitial, IncludeAll

using Distributions
using Rocket

import Rocket: getrecent
import Base: ndims, precision, length, size

struct Marginal{D}
    data       :: D
    is_clamped :: Bool
    is_initial :: Bool
end

Base.show(io::IO, marginal::Marginal) = print(io, string("Marginal(", getdata(marginal), ")"))

getdata(marginal::Marginal)    = marginal.data
is_clamped(marginal::Marginal) = marginal.is_clamped
is_initial(marginal::Marginal) = marginal.is_initial

# TupleTools.prod is a more efficient version of Base.all for NTuple here
is_clamped(marginals::Tuple)  = TupleTools.prod(map(is_clamped, marginals))
is_initial(marginals::Tuple)  = TupleTools.prod(map(is_initial, marginals))

## Statistics 

Distributions.pdf(marginal::Marginal, x)    = Distributions.pdf(getdata(marginal), x)
Distributions.logpdf(marginal::Marginal, x) = Distributions.logpdf(getdata(marginal), x)

MacroHelpers.@proxy_methods Marginal getdata [
    Distributions.mean,
    Distributions.median,
    Distributions.mode,
    Distributions.shape,
    Distributions.scale,
    Distributions.rate,
    Distributions.var,
    Distributions.std,
    Distributions.cov,
    Distributions.invcov,
    Distributions.logdetcov,
    Distributions.entropy,
    Distributions.params,
    Base.precision,
    Base.length,
    Base.ndims,
    Base.size,
    mean_cov, 
    mean_invcov, 
    mean_precision, 
    weightedmean_cov, 
    weightedmean_invcov, 
    weightedmean_precision,
    probvec,
    weightedmean,
    inversemean,
    logmean,
    meanlogmean,
    mirroredlogmean,
    loggammamean
]

## Utility functions

getdata(marginals::Tuple) = map(getdata, marginals)

as_marginal(marginal::Marginal) = marginal

## Marginal observable

abstract type MarginalSkipStrategy end

struct SkipClamped           <: MarginalSkipStrategy end
struct SkipInitial           <: MarginalSkipStrategy end
struct SkipClampedAndInitial <: MarginalSkipStrategy end
struct IncludeAll            <: MarginalSkipStrategy end

apply_skip_filter(observable, ::SkipClamped)           = observable |> filter(v -> !is_clamped(v))
apply_skip_filter(observable, ::SkipInitial)           = observable |> filter(v -> !is_initial(v))
apply_skip_filter(observable, ::SkipClampedAndInitial) = observable |> filter(v -> !is_initial(v) && !is_clamped(v))
apply_skip_filter(observable, ::IncludeAll)            = observable

struct MarginalObservable <: Subscribable{ Marginal }
    subject :: Rocket.RecentSubjectInstance{ Marginal, Subject{ Marginal, AsapScheduler, AsapScheduler } }
    stream  :: LazyObservable{ Marginal }
end

MarginalObservable() = MarginalObservable(RecentSubject(Marginal), lazy(Marginal))   

as_marginal_observable(observable::MarginalObservable, skip_strategy::MarginalSkipStrategy) = apply_skip_filter(observable, skip_strategy)
as_marginal_observable(observable)                                                          = as_marginal_observable(observable, IncludeAll())

function as_marginal_observable(observable, skip_strategy::MarginalSkipStrategy)
    output = MarginalObservable()
    connect!(output, observable)
    return as_marginal_observable(output, skip_strategy)
end

Rocket.getrecent(observable::MarginalObservable) = Rocket.getrecent(observable.subject)
Rocket.getrecent(observables::Tuple)             = Rocket.getrecent.(observables)
Rocket.getrecent(::Nothing)                      = nothing

@inline Rocket.on_subscribe!(observable::MarginalObservable, actor) = subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Actor{ <: Marginal })           = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.NextActor{ <: Marginal })       = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ErrorActor{ <: Marginal })      = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.CompletionActor{ <: Marginal }) = Rocket.on_subscribe!(observable.stream, actor)

@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.Subject{ <: Marginal })                 = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.BehaviorSubjectInstance{ <: Marginal }) = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.PendingSubjectInstance{ <: Marginal })  = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.RecentSubjectInstance{ <: Marginal })   = Rocket.on_subscribe!(observable.stream, actor)
@inline Rocket.subscribe!(observable::MarginalObservable, actor::Rocket.ReplaySubjectInstance{ <: Marginal })   = Rocket.on_subscribe!(observable.stream, actor)

function connect!(marginal::MarginalObservable, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function setmarginal!(marginal::MarginalObservable, value)
    next!(marginal.subject, Marginal(value, false, true))
    return nothing
end