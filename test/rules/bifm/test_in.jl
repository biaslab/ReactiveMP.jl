module RulesBIFMInTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:BIFM:in" begin

     @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta)" begin

        meta = BIFMMeta([2 0; 0 1], # A
                        [3 0; 0 2], # B
                        [4 0; 0 3], # C
                        [5 0; 0 4], # H
                        [6 0; 0 5], # BHBt
                        [1, 2],     # ξz
                        [7 0; 0 6], # Λz
                        [2, 3],     # ξztilde
                        [8 0; 0 7], # Λztilde
                        [3, 4],     # μu
                        [9 0; 0 8]  # Σu
                        )

        @test_rules [ with_float_conversions = false ] BIFM(:in, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([1,2], [2 0; 0 1]),     m_zprev = MarginalDistribution(MvNormalMeanPrecision([1,2], [1 0; 0 2])), m_znext = MvNormalMeanPrecision([1,2], [1 0; 0 2]), meta = meta),
                output = MarginalDistribution(MvNormalMeanCovariance([-375, -172], [180801 0.0; 0.0 4487.999999999998]))),
            (input = (m_out = MvNormalMeanPrecision([3,4], [6 0; 0 1]),     m_zprev = MarginalDistribution(MvNormalMeanPrecision([1,6], [2 0; 0 2])), m_znext = MvNormalMeanPrecision([8,2], [2 0; 0 2]), meta = meta),
                output = MarginalDistribution(MvNormalMeanCovariance([-375, -620], [87488.99999999997 0.0; 0.0 4487.999999999998]))),
            (input = (m_out = MvNormalMeanPrecision([5,6], [2 0; 0 5]),     m_zprev = MarginalDistribution(MvNormalMeanPrecision([6,2], [1 0; 0 1])), m_znext = MvNormalMeanPrecision([1,9], [1 0; 0 1]), meta = meta),
                output = MarginalDistribution(MvNormalMeanCovariance([-2535, -172], [180801 0.0; 0.0 10760.0])))
        ]

    end

end

end