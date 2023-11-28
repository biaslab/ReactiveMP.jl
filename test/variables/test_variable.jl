module ReactiveMPVariableTest

using Test, ReactiveMP, Rocket, BayesBase, Distributions, ExponentialFamily

struct CustomDeterministicNode end

@node CustomDeterministicNode Deterministic [out, (x, aliases = [xx])]

function test_variable_set_method(variable, dist::T, k) where {T}
    flag = false

    activate!(variable, TestOptions())

    test_out_var = randomvar(:out)

    # messages could be initialized only when the node is created
    for _ in 1:k
        make_node(identity, ReactiveMP.FactorNodeCreationOptions(ReactiveMP.DeltaFn, TestNodeMetaData(), nothing), test_out_var, variable)
    end

    for node_index in 1:k
        setmessage!(variable, node_index, dist)
    end
    
    setmarginal!(variable, dist)

    subscription = subscribe!(getmarginal(variable, IncludeAll()), (marginal) -> begin
        @test typeof(marginal) <: Marginal{T}
        @test mean(marginal) === mean(dist)
        @test var(marginal) === var(dist)
        flag = true
    end)

    for node_index in 1:k
        subscription = subscribe!(ReactiveMP.messageout(variable, node_index), (message) -> begin
            @test typeof(message) <: Message{T}
            @test mean(message) === mean(dist)
            @test var(message) === var(dist)
        end)
    end


    # Test that subscription happenend
    @test flag === true

    unsubscribe!(subscription)
end

struct TestNodeMetaData end

ReactiveMP.collect_meta(::Type{D}, options::FactorNodeCreationOptions{F, T}) where {D <: DeltaFn, F, T <: TestNodeMetaData} = TestNodeMetaData()
ReactiveMP.getinverse(::TestNodeMetaData) = nothing
ReactiveMP.getinverse(::TestNodeMetaData, k::Int) = nothing

function test_variables_set_methods(variables, dist::T, k::Int) where {T}
    marginal_subscription_flag = false

    activate!.(variables, TestOptions())

    @test_throws AssertionError setmarginals!(variables, Iterators.repeated(dist, length(variables) - 1))

    test_out_var = randomvar(:out)
    
    for _ in 1:k
        make_node(identity, ReactiveMP.FactorNodeCreationOptions(ReactiveMP.DeltaFn, TestNodeMetaData(), nothing), test_out_var, variables...)
    end

    @test all(degree.(variables) .== k)

    @test_throws AssertionError setmessages!(variables, Iterators.repeated(dist, length(variables) - 1))

    setmarginals!(variables, dist)

    @test_throws AssertionError setmessages!(variables, Iterators.repeated(dist, length(variables) - 1))

    setmessages!(variables, dist)

    subscription = subscribe!(getmarginals(variables, IncludeAll()) |> take(1), (marginals) -> begin
        @test length(marginals) === length(variables)
        foreach(marginals) do marginal
            @test typeof(marginal) <: Marginal{T}
            @test mean(marginal) === mean(dist)
            @test var(marginal) === var(dist)
        end
        marginal_subscription_flag = true
    end)

    # Test that subscription happenend
    @test marginal_subscription_flag === true

    unsubscribe!(subscription)

    for node_index in 1:k

        subscription = subscribe!(collectLatest(ReactiveMP.messageout.(variables, node_index)) |> take(1), (messages) -> begin
            @test length(messages) === length(variables)
            foreach(messages) do message
                @test typeof(message) <: Message{T}
                @test mean(message) === mean(dist)
                @test var(message) === var(dist)
            end
        end)

        unsubscribe!(subscription)
    
    end
end

@testset "Variable" begin
    import ReactiveMP: activate!
    import Rocket: getscheduler

    struct TestOptions end

    Rocket.getscheduler(::TestOptions) = AsapScheduler()
    Base.broadcastable(::TestOptions) = Ref(TestOptions()) # for broadcasting

    @testset "setmarginal! and setmessages! tests for randomvar" begin
        dists = (NormalMeanVariance(-2.0, 3.0), NormalMeanPrecision(-2.0, 3.0), PointMass(2.0))
        number_of_nodes = 1:4
        for (dist, k) in Iterators.product(dists, number_of_nodes)
            test_variable_set_method(randomvar(:r), dist, k)
            test_variables_set_methods(randomvar(:r, 2), dist, k)
            test_variables_set_methods(randomvar(:r, 2, 2), dist, k)
        end
    end
end

end
