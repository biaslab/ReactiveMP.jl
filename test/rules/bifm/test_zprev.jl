module RulesBIFMZprevTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:BIFM:zprev" begin

     @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta)" begin

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

        @test_rules [ with_float_conversions = false ] BIFM(:zprev, Marginalisation) [
            (input = (m_out = MvNormalMeanPrecision([1,2], [2 0; 0 1]),     m_in = MvNormalMeanPrecision([1,2], [1 0; 0 2]), m_znext = MvNormalMeanPrecision([1,2], [1 0; 0 2]), meta = meta),
                output = MvNormalWeightedMeanPrecision([-0.60402684563758, -1.4782608695652169], [0.44295302013420557 0.0; 0.0 0.4782608695652171])),
            (input = (m_out = MvNormalMeanPrecision([3,4], [6 0; 0 1]),     m_in = MvNormalMeanPrecision([1,6], [2 0; 0 2]), m_znext = MvNormalMeanPrecision([8,2], [2 0; 0 2]), meta = meta),
                output = MvNormalWeightedMeanPrecision([-0.9321266968325688, -5.043478260869566],[0.886877828054339 0.0; 0.0 0.4782608695652171])),
            (input = (m_out = MvNormalMeanPrecision([5,6], [2 0; 0 5]),     m_in = MvNormalMeanPrecision([6,2], [1 0; 0 1]), m_znext = MvNormalMeanPrecision([1,9], [1 0; 0 1]), meta = meta),
                output = MvNormalWeightedMeanPrecision([-3.7114093959731633, -0.45945945945943834], [0.44295302013420557 0.0; 0.0 0.24864864864865122]))
        ]

    end

end

end