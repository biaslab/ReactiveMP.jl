module RulesSubtractionMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_marginalrules

@testset "marginalrules:Subtraction" begin

    @testset ":in1_in2 (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin
        
        @test_marginalrules [ with_float_conversions = false ] typeof(-)(:in1_in2) [
            (
                input = (m_out = NormalMeanVariance(3, 4), m_in1 = NormalMeanVariance(2, 2), m_in2 = PointMass(2.0)), 
                output = (in1 = NormalWeightedMeanPrecision(9/4, 3/4), in2 = PointMass(2.0))
            ),

        ]


    end
end
end