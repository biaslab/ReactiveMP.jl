module BIFMHelperNodeTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "BIFMHelperNode" begin

    @testset "Creation" begin
        node = make_node(BIFMHelper)

        @test functionalform(node)          === BIFMHelper
        @test sdtype(node)                  === Stochastic()
        @test name.(interfaces(node))       === (:out, :in)
        @test factorisation(node)           === ((1, 2), )

        @test length(methods(functional_dependencies, (Any, FactorNode{Type{BIFMHelper}}, Int))) === 1

    end

    @testset "Average energy" begin
        
        node = make_node(BIFMHelper)

        @test score(AverageEnergy(),
                    BIFMHelper,
                    Val{(:out, :in)},
                    (Marginal(MvNormalMeanCovariance([1,1], [2 0; 0 3]), false, false), Marginal(MvNormalMeanCovariance([1,1], [2 0; 0 3]), false, false)),
                    nothing) ≈ entropy(MvNormalMeanCovariance([1,1], [2 0; 0 3]))
        
        @test score(AverageEnergy(),
                    BIFMHelper,
                    Val{(:out, :in)},
                    (Marginal(MvNormalMeanCovariance([1,2], [2 0; 0 1]), false, false), Marginal(MvNormalMeanPrecision([1,2], [0.5 0; 0 1]), false, false)),
                    nothing) ≈ entropy(MvNormalMeanCovariance([1,2], [2 0; 0 1]))
                    
    end

end

end