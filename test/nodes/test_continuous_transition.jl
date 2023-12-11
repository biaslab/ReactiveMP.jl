module ContinuousTransitionNodeTest

using Test, ReactiveMP, Random, Distributions, BayesBase, ExponentialFamily

import ReactiveMP: getdimensionality, getjacobians, gettransformation, getunits, ctcompanion_matrix

@testset "ContinuousTransitionNode" begin
    dy, dx = 2, 3
    meta = CTMeta(a -> reshape(a, dy, dx))
    @testset "Creation" begin
        node = make_node(ContinuousTransition, FactorNodeCreationOptions(nothing, meta, nothing))

        @test functionalform(node) === ContinuousTransition
        @test sdtype(node) === Stochastic()
        @test name.(interfaces(node)) === (:y, :x, :a, :W)
        @test factorisation(node) === ((1, 2, 3, 4),)
    end

    @testset "AverageEnergy" begin
        q_y_x = MvNormalMeanCovariance(zeros(5), diageye(5))
        q_a = MvNormalMeanCovariance(zeros(6), diageye(6))
        q_W = Wishart(3, diageye(2))

        marginals = (Marginal(q_y_x, false, false, nothing), Marginal(q_a, false, false, nothing), Marginal(q_W, false, false, nothing))

        @test score(AverageEnergy(), ContinuousTransition, Val{(:y_x, :a, :W)}(), marginals, meta) ≈ 13.415092731310878
        @show getjacobians(meta, mean(q_a))
    end

    @testset "ContinuousTransition Functionality" begin
        m_a = randn(6)
        A = ctcompanion_matrix(m_a, zeros(length(m_a)), meta)

        @test size(A) == (dy, dx)
        @test A == gettransformation(meta)(m_a)
    end
end

end
