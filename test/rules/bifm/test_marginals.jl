module RulesBIFMMarginalsTest

using Test
using ReactiveMP
import ReactiveMP: @test_marginalrules

@testset "marginalrules:BIFM" begin

    @testset ":in_zprev_znext (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily)" begin

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

        @test_marginalrules [ with_float_conversions = false ] BIFM(:in_zprev_znext) [
            (
                input  = (m_out = MvNormalMeanPrecision([1,2], [2 0; 0 1]),     m_in = MvNormalMeanPrecision([1,2], [1 0; 0 2]), m_zprev = MarginalDistribution(MvNormalMeanPrecision([1,2], [1 0; 0 2])), m_znext = MvNormalMeanPrecision([1,2], [1 0; 0 3]), meta = meta), 
                output = MvNormalWeightedMeanPrecision([28.0, 28.0, 15.0, 13.0], [298.0 0.0 198.0 0.0; 0.0 50.0 0.0 24.0; 198.0 0.0 101.0 0.0; 0.0 24.0 0.0 7.0])
            ),
            (
                input  = (m_out = MvNormalMeanPrecision([3,4], [6 0; 0 1]),     m_in = MvNormalMeanPrecision([8,2], [2 0; 0 2]), m_zprev = MarginalDistribution(MvNormalMeanPrecision([1,6], [2 0; 0 2])), m_znext = MvNormalMeanPrecision([1,2], [1 0; 0 3]), meta = meta), 
                output = MvNormalWeightedMeanPrecision([235.0, 40.0, 144.0, 27.0], [875.0 0.0 582.0 0.0; 0.0 50.0 0.0 24.0; 582.0 0.0 358.0 0.0; 0.0 24.0 0.0 7.0])
            ),
            (
                input  = (m_out = MvNormalMeanPrecision([5,6], [2 0; 0 5]),     m_in = MvNormalMeanPrecision([1,9], [1 0; 0 1]), m_zprev = MarginalDistribution(MvNormalMeanPrecision([6,2], [1 0; 0 1])), m_znext = MvNormalMeanPrecision([1,2], [1 0; 0 3]), meta = meta), 
                output = MvNormalWeightedMeanPrecision([124.0, 201.0, 84.0, 95.0], [298.0 0.0 198.0 0.0; 0.0 193.0 0.0 96.0; 198.0 0.0 101.0 0.0; 0.0 96.0 0.0 42.0])
            ),

        ]


    end
end

end