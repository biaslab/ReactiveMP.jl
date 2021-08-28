module RulesAdditionMarginalsTest

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_marginalrules

@testset "marginalrules:Addition" begin

    @testset ":in1_in2 (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin
        
        @test_marginalrules [ with_float_conversions = true ] typeof(+)(:in1_in2) [
            (
                input = (m_out = NormalMeanVariance(3.0, 4.0), m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(2.0)),
                output = (in1 = NormalWeightedMeanPrecision(5/4, 3/4), in2 = PointMass(2.0))
            ),
            (
                input = (m_out = NormalMeanPrecision(3, 4), m_in1 = NormalMeanPrecision(1, 4), m_in2 = PointMass(1.0)),
                output = (in1 = NormalWeightedMeanPrecision(12.0, 8.0), in2 = PointMass(1.0))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(3.0, 4.0), m_in1 = NormalWeightedMeanPrecision(1.0, 4.0), m_in2 = PointMass(1.0)),
                output = (in1 = NormalWeightedMeanPrecision(0.0, 8.0), in2 = PointMass(1.0))
            ),
            (
                input = (m_out = NormalMeanPrecision(2.0, 4.0), m_in1 = NormalMeanVariance(1.0, 2.0), m_in2 = PointMass(1.0)),
                output = (in1 = NormalWeightedMeanPrecision(9/2, 9/2), in2 = PointMass(1.0))
            ),
            (
                input = (m_out = NormalMeanVariance(2.0, 4.0), m_in1 = NormalMeanPrecision(1.0, 2.0), m_in2 = PointMass(1.0)),
                output = (in1 = NormalWeightedMeanPrecision(9/4, 9/4), in2 = PointMass(1.0))
            ),
            (
                input = (m_out = NormalMeanPrecision(3.0, 4.0), m_in1 = NormalWeightedMeanPrecision(1.0, 2.0), m_in2 = PointMass(2.0)),
                output = (in1 = NormalWeightedMeanPrecision(5.0, 6.0), in2 = PointMass(2.0))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 2.0), m_in1 = NormalMeanPrecision(1.0, 2.0), m_in2 = PointMass(-1.0)),
                output = (in1 = NormalWeightedMeanPrecision(6.0, 4.0), in2 = PointMass(-1.0))
            ),
            (
                input = (m_out = NormalMeanVariance(3.0, 3.0), m_in1 = NormalWeightedMeanPrecision(2.0, 1.0), m_in2 = PointMass(2.0)),
                output = (in1 = NormalWeightedMeanPrecision(7/3, 4/3), in2 = PointMass(2.0))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 4.0), m_in1 = NormalMeanVariance(2.0, 2.0), m_in2 = PointMass(1.0)),
                output = (in1 = NormalWeightedMeanPrecision(-1.0, 9/2), in2 = PointMass(1.0))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(+)(:in1_in2) [
            (
                input = (m_out = NormalMeanVariance(3.0, 4.0), m_in1 = PointMass(2.0), m_in2 = NormalMeanVariance(2.0, 2.0)),
                output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(5/4, 3/4))
            ),
            (
                input = (m_out = NormalMeanPrecision(3, 4), m_in1 = PointMass(1.0), m_in2 = NormalMeanPrecision(1, 4)),
                output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(12.0, 8.0))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(3.0, 4.0), m_in1 = PointMass(1.0), m_in2 = NormalWeightedMeanPrecision(1.0, 4.0)),
                output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(0.0, 8.0))
            ),
            (
                input = (m_out = NormalMeanPrecision(2.0, 4.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(1.0, 2.0)),
                output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(9/2, 9/2))
            ),
            (
                input = (m_out = NormalMeanVariance(2.0, 4.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanPrecision(1.0, 2.0)),
                output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(9/4, 9/4))
            ),
            (
                input = (m_out = NormalMeanPrecision(3.0, 4.0), m_in1 = PointMass(2.0), m_in2 = NormalWeightedMeanPrecision(1.0, 2.0)),
                output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(5.0, 6.0))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 2.0), m_in1 = PointMass(-1.0), m_in2 = NormalMeanPrecision(1.0, 2.0)),
                output = (in1 = PointMass(-1.0), in2 = NormalWeightedMeanPrecision(6.0, 4.0))
            ),
            (
                input = (m_out = NormalMeanVariance(3.0, 3.0), m_in1 = PointMass(2.0), m_in2 = NormalWeightedMeanPrecision(2.0, 1.0)),
                output = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(7/3, 4/3))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(2.0, 4.0), m_in1 = PointMass(1.0), m_in2 = NormalMeanVariance(2.0, 2.0)),
                output = (in1 = PointMass(1.0), in2 = NormalWeightedMeanPrecision(-1.0, 9/2))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::MvNormalMeanCovariance, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_marginalrules [ with_float_conversions = false ] typeof(+)(:in1_in2) [
            (
                input = (m_out = MvNormalMeanCovariance([2.0, 2.0], [2.0 0.0; 0.0 2.0]), m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), m_in2 = PointMass([1.0, 1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([1/4, 11/8], [1.0 -1/4; -1/4 7/8]), in2 = PointMass([1.0, 1.0]))
            ),
            (
                input = (m_out = MvNormalMeanCovariance([3.0, 2.0], [2.0 0.0; 0.0 2.0]), m_in1 = MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]), m_in2 = PointMass([1.0, 1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([4.0, 7/2], [5/2 1.0; 1.0 5/2]), in2 = PointMass([1.0, 1.0]))
            ),
            (
                input = (m_out = MvNormalMeanCovariance([3.0, 2.0], [4.0 2.0; 2.0 4.0]), m_in1 = MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 0.0; 0.0 2.0]), m_in2 = PointMass([2.0, 1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([7/6, 13/6], [7/3 -1/6; -1/6 7/3]), in2 = PointMass([2.0, 1.0]))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::MvNormalMeanPrecision, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(+)(:in1_in2) [
            (
                input = (m_out = MvNormalMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), m_in1 = MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0]), m_in2 = PointMass([1.0, -1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([5/4, 29/8], [3/2 -1/4; -1/4 11/8]), in2 = PointMass([1.0, -1.0]))
            ),
            (
                input = (m_out = MvNormalMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 3.0]), m_in1 = MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]), m_in2 = PointMass([2.0, 2.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([-1.0, -1.0], [5.0 2.0; 2.0 5.0]), in2 = PointMass([2.0, 2.0]))
            ),
            (
                input = (m_out = MvNormalMeanPrecision([3.0, 2.0], [4.0 2.0; 2.0 4.0]), m_in1 = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]), m_in2 = PointMass([-2.0, 1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([23.0, 16.0], [7.0  3.0; 3.0 5.0]), in2 = PointMass([-2.0, 1.0]))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::MvNormalWeightedMeanPrecision, m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_marginalrules [ with_float_conversions = false ] typeof(+)(:in1_in2) [
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), m_in1 = MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0]), m_in2 = PointMass([1.0, -1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([5/4, 29/8], [3/2 -1/4; -1/4 11/8]), in2 = PointMass([1.0, -1.0]))
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]), m_in1 = MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]), m_in2 = PointMass([2.0, 2.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([-4.0, -2.0], [5.0 2.0; 2.0 4.0]), in2 = PointMass([2.0, 2.0]))
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([1.0, 2.0], [4.0 1.0; 1.0 4.0]), m_in1 = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 0.0; 0.0 2.0]), m_in2 = PointMass([-1.0, 1.0])),
                output = (in1 = MvNormalWeightedMeanPrecision([5.0, 0.0], [7.0  1.0; 1.0 6.0]), in2 = PointMass([-1.0, 1.0]))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::MvNormalMeanCovariance, m_in1::PointMass, m_in2::PointMassMultivariateNormalDistributionsFamily)" begin

        @test_marginalrules [ with_float_conversions = false ] typeof(+)(:in1_in2) [
            (
                input = (m_out = MvNormalMeanCovariance([2.0, 2.0], [2.0 0.0; 0.0 2.0]), m_in1 = PointMass([1.0, 1.0]), m_in2 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),
                output = (in1 = PointMass([1.0, 1.0]), in2 = MvNormalWeightedMeanPrecision([1/4, 11/8], [1.0 -1/4; -1/4 7/8]))
            ),
            (
                input = (m_out = MvNormalMeanCovariance([3.0, 2.0], [2.0 0.0; 0.0 2.0]), m_in1 = PointMass([1.0, 1.0]), m_in2 = MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0])),
                output = (in1 = PointMass([1.0, 1.0]), in2 = MvNormalWeightedMeanPrecision([4.0, 7/2], [5/2 1.0; 1.0 5/2]))
            ),
            (
                input = (m_out = MvNormalMeanCovariance([3.0, 2.0], [4.0 2.0; 2.0 4.0]), m_in1 = PointMass([2.0, 1.0]), m_in2 = MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 0.0; 0.0 2.0])),
                output = (in1 = PointMass([2.0, 1.0]), in2 = MvNormalWeightedMeanPrecision([7/6, 13/6], [7/3 -1/6; -1/6 7/3]))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::MvNormalMeanPrecision, m_in1::, m_in2::MultivariateNormalDistributionsFamilyPointMass)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(+)(:in1_in2) [
            (
                input = (m_out = MvNormalMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), m_in1 = PointMass([1.0, -1.0]), m_in2 = MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0])),
                output = (in1 = PointMass([1.0, -1.0]), in2 = MvNormalWeightedMeanPrecision([5/4, 29/8], [3/2 -1/4; -1/4 11/8]))
            ),
            (
                input = (m_out = MvNormalMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 3.0]), m_in1 = PointMass([2.0, 2.0]), m_in2 = MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0])),
                output = (in1 = PointMass([2.0, 2.0]), in2 = MvNormalWeightedMeanPrecision([-1.0, -1.0], [5.0 2.0; 2.0 5.0]))
            ),
            (
                input = (m_out = MvNormalMeanPrecision([3.0, 2.0], [4.0 2.0; 2.0 4.0]), m_in1 = PointMass([-2.0, 1.0]), m_in2 = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0])),
                output = (in1 = PointMass([-2.0, 1.0]), in2 = MvNormalWeightedMeanPrecision([23.0, 16.0], [7.0  3.0; 3.0 5.0]))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::MvNormalWeightedMeanPrecision, m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily)" begin

        @test_marginalrules [ with_float_conversions = true ] typeof(+)(:in1_in2) [
            (
                input = (m_out = MvNormalWeightedMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), m_in1 = PointMass([1.0, -1.0]), m_in2 = MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0])),
                output = (in1 = PointMass([1.0, -1.0]), in2 = MvNormalWeightedMeanPrecision([5/4, 29/8], [3/2 -1/4; -1/4 11/8]))
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]), m_in1 = PointMass([2.0, 2.0]), m_in2 = MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0])),
                output = (in1 = PointMass([2.0, 2.0]), in2 = MvNormalWeightedMeanPrecision([-4.0, -2.0], [5.0 2.0; 2.0 4.0]))
            ),
            (
                input = (m_out = MvNormalWeightedMeanPrecision([1.0, 2.0], [4.0 1.0; 1.0 4.0]), m_in1 = PointMass([-1.0, 1.0]), m_in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 0.0; 0.0 2.0])),
                output = (in1 = PointMass([-1.0, 1.0]), in2 = MvNormalWeightedMeanPrecision([5.0, 0.0], [7.0  1.0; 1.0 6.0]))
            ),

        ]

    end

    @testset ":in1_in2 (m_out::NormalDistributionsFamily, m_in1::NormalDistributionsFamily, m_in2::NormalDistributionsFamily)" begin

        @test_marginalrules [ with_float_conversions = false ] typeof(+)(:in1_in2) [
            (
                input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = NormalMeanVariance(3.0, 4.0), m_in2 = NormalMeanVariance(5.0, 6.0)),
                output = (MvNormalWeightedMeanPrecision([5/4, 4/3], [3/4 1/2; 1/2 2/3]))
            ),
            (
                input = (m_out = NormalMeanPrecision(1.0, 2.0), m_in1 = NormalMeanPrecision(3.0, 4.0), m_in2 = NormalMeanPrecision(5.0, 6.0)),
                output = (MvNormalWeightedMeanPrecision([14.0, 32.0], [6.0 2.0; 2.0 8.0]))
            ),
            (
                input = (m_out = NormalWeightedMeanPrecision(1.0, 2.0), m_in1 = NormalWeightedMeanPrecision(3.0, 4.0), m_in2 = NormalWeightedMeanPrecision(5.0, 6.0)),
                output = (MvNormalWeightedMeanPrecision([4.0, 6.0], [6.0 2.0; 2.0 8.0]))
            ),
            (
                input = (m_out = NormalMeanVariance(1.0, 2.0), m_in1 = NormalMeanPrecision(3.0, 4.0), m_in2 = NormalWeightedMeanPrecision(5.0, 6.0)),
                output = (MvNormalWeightedMeanPrecision([25/2, 11/2], [9/2 1/2; 1/2 13/2]))
            ),
            (
                input = (m_out = MvNormalMeanCovariance([1.0, 2.0], [3.0 1.0; 1.0 2.0]), m_in1 = MvNormalMeanCovariance([2.0, 3.0], [3.0 1.0; 1.0 2.0]), m_in2 = MvNormalMeanCovariance([1.0, 2.0], [3.0 1.0; 1.0 2.0])),
                output = (MvNormalWeightedMeanPrecision([[1/5, 12/5]; [0.0, 2.0]], [[0.8 -0.4 -0.4 1.2; 0.4 -0.2 -0.2 0.6]; [0.4 -0.2 -0.2 0.6; 0.8 -0.4 -0.4 1.2]]))
            ),
            #(
            #    input = (m_out = MvNormalMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 2.0]), m_in1 = MvNormalMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 2.0]), m_in2 = MvNormalMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 2.0])),
            #    output = (MvNormalWeightedMeanPrecision([[1/5, 12/5]; [0.0, 2.0]], [6.0 2.0; 2.0 8.0]))
            #),
            #(
            #    input = (m_out = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]), m_in1 = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]), m_in2 = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0])),
            #    output = (MvNormalWeightedMeanPrecision([[4.0, 6.0]; [4.0, 6.0]], [6.0 2.0; 2.0 8.0]))
            #),
            #(
            #    input = (m_out = MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 2.0]), m_in1 = MvNormalMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]), m_in2 = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0])),
            #    output = (MvNormalWeightedMeanPrecision([[4.0, 6.0]; [4.0, 6.0]], [6.0 2.0; 2.0 8.0]))
            #),

        ]

    end

end
end