module RulesBIFMNextTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:BIFM:znext" begin

     @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, meta::BIFMMeta)" begin

        meta = BIFMMeta([2.0 0; 0 1], # A
                        [3.0 0; 0 2], # B
                        [4.0 0; 0 3], # C
                        [5.0 0; 0 4], # H
                        [6.0 0; 0 5], # BHBt
                        [1.0, 2],     # ξz
                        [7.0 0; 0 6], # Λz
                        [2.0, 3],     # ξztilde
                        [8.0 0; 0 7], # Λztilde
                        [3.0, 4],     # μu
                        [9.0 0; 0 8]  # Σu
                        )

        @test_rules [ with_float_conversions = false ] BIFM(:znext, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([1,2], [2 0; 0 1]),     m_in = MvNormalMeanPrecision([1,2], [1 0; 0 2]), m_zprev = MarginalDistribution(MvNormalMeanPrecision([1,2], [1 0; 0 2])), meta = meta),
                output = MarginalDistribution(MvNormalMeanCovariance([-61.0, -48.0], [6730.0 0.0; 0.0 425.5]))),
            (input = (m_out = MvNormalMeanPrecision([3,4], [6 0; 0 1]),     m_in = MvNormalMeanPrecision([8,2], [2 0; 0 2]), m_zprev = MarginalDistribution(MvNormalMeanPrecision([1,6], [2 0; 0 2])), meta = meta),
                output = MarginalDistribution(MvNormalMeanCovariance([-49, -164], [3368 0.0; 0.0 425.5]))),
            (input = (m_out = MvNormalMeanPrecision([5,6], [2 0; 0 5]),     m_in = MvNormalMeanPrecision([1,9], [1 0; 0 1]), m_zprev = MarginalDistribution(MvNormalMeanPrecision([6,2], [1 0; 0 1])), meta = meta),
                output = MarginalDistribution(MvNormalMeanCovariance([-471.0, -28.0], [6730.0 0.0; 0.0 846.0])))
        ]

    end

end

end