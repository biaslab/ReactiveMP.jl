module RulesSubtractionIn2Test

using Test
using ReactiveMP
using Random
import ReactiveMP: @test_rules

@testset "rules:typeof(-):in2" begin

     @testset "Belief Propagation: (m_out::PointMass, m_in2::PointMass)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = PointMass(1.0),                    m_in1 = PointMass(-1.0)),                    output = PointMass(-2.0)),
            (input = (m_out = PointMass([ 1.0 ]),                m_in1 = PointMass([ -2.0 ])),                output = PointMass([ -3.0 ])),
            (input = (m_out = PointMass([ 1.0 2.0; 3.0 4.0 ]),   m_in1 = PointMass([-2.0 -1.0; -4.0 -1.0])),  output = PointMass([ -3.0 -3.0; -7.0 -5.0 ])),
        ]

    end

     @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0),   m_in1 = PointMass(2.0)),  output = NormalMeanVariance(1.0, 2.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 3.0),  m_in1 = PointMass(-3.0)), output = NormalMeanVariance(-2.0, 3.0)),
            (input = (m_out = NormalMeanPrecision(4.0, 7.0),  m_in1 = PointMass(1.0)),  output = NormalMeanPrecision(-3.0, 7.0)),
            (input = (m_out = NormalMeanPrecision(-1.0, 2.0), m_in1 = PointMass(-3.0)), output = NormalMeanPrecision(-2.0, 2.0)),
            (input = (m_out = NormalWeightedMeanPrecision(4.0, 2.0), m_in1 = PointMass(1.0)),  output = NormalMeanVariance(-1.0, 1/2)),
            (input = (m_out = NormalWeightedMeanPrecision(8.0, 4.0), m_in1 = PointMass(-3.0)), output = NormalMeanVariance(-5.0, 1/4)),
        ]

    end

        @testset "Belief Propagation: (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = PointMass(1.0),  m_in1 = NormalMeanVariance(1.0, 2.0)),   output = NormalMeanVariance(0.0, 2.0)),
            (input = (m_out = PointMass(-3.0), m_in1 = NormalMeanVariance(-1.0, 3.0)),  output = NormalMeanVariance(2.0, 3.0)),
            (input = (m_out = PointMass(1.0),  m_in1 = NormalMeanPrecision(4.0, 7.0)),  output = NormalMeanPrecision(3.0, 7.0)),
            (input = (m_out = PointMass(-3.0), m_in1 = NormalMeanPrecision(-1.0, 2.0)), output = NormalMeanPrecision(2.0, 2.0)),
            (input = (m_out = PointMass(1.0),  m_in1 = NormalWeightedMeanPrecision(4.0, 2.0)),  output = NormalMeanVariance(1.0, 1/2)),
            (input = (m_out = PointMass(-3.0), m_in1 = NormalWeightedMeanPrecision(8.0, 4.0)),  output = NormalMeanVariance(5.0, 1/4)),
        ]

    end

        @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_in2::PointMass)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([1.0, 1.0])),  output = MvNormalMeanCovariance([0.0, -2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), m_in1 = PointMass([1.0, 1.0])),  output = MvNormalMeanCovariance([5.0, -2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([3.0, 2.0])),  output = MvNormalMeanPrecision([7.0, -1.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]),  m_in1 = PointMass([-2.0, 3.0])), output = MvNormalMeanPrecision([2.0, 0.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = MvNormalWeightedMeanPrecision([1.0, 3.0], [ 1.0 0.0; 0.0 1.0 ]), m_in1 = PointMass([ 2.0, 7.0 ])),  output = MvNormalMeanCovariance([ 1.0, 4.0 ], [ 1.0 0.0; 0.0 1.0 ])),
            (input = (m_out = MvNormalWeightedMeanPrecision([2.0, 4.0], [ 2.0 0.0; 0.0 2.0 ]), m_in1 = PointMass([ 1.0, 3.0 ])),  output = MvNormalMeanCovariance([ 0.0, 1.0 ], [ 1/2 0.0; 0.0 1/2 ])),
        ]

    end

    @testset "Belief Propagation: (m_out::PointMass, m_in2::MultivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = PointMass([1.0, 1.0]),    m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),    output = MvNormalMeanCovariance([0.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = PointMass([1.0, 1.0]),    m_in1 = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0])),   output = MvNormalMeanCovariance([-5.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = PointMass([1.0, 1.0]),    m_in1 = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0])),    output = MvNormalMeanPrecision([-5.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = PointMass([-2.0, 1.0]),   m_in1 = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0])),    output = MvNormalMeanPrecision([-2.0, 2.0], [3.0 2.0; 2.0 4.0])),
            (input = (m_out = PointMass([ 2.0, 7.0 ]),  m_in1 = MvNormalWeightedMeanPrecision([1.0, 3.0], [ 1.0 0.0; 0.0 1.0 ])),  output = MvNormalMeanCovariance([ -1.0, -4.0 ], [ 1.0 0.0; 0.0 1.0 ])),
            (input = (m_out = PointMass([ 1.0, 3.0 ]),  m_in1 = MvNormalWeightedMeanPrecision([2.0, 4.0], [ 2.0 0.0; 0.0 2.0 ])),  output = MvNormalMeanCovariance([ 0.0, -1.0 ], [ 1/2 0.0; 0.0 1/2 ])),
        ]

    end

    @testset "Belief Propagation: (m_out::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = NormalMeanVariance(1.0, 2.0),             m_in1 = NormalMeanVariance(3.0, 4.0)),          output = NormalMeanVariance(2.0, 6.0)),
            (input = (m_out = NormalMeanVariance(-1.0, 2.0),            m_in1 = NormalMeanVariance(-2.0, 3.0)),         output = NormalMeanVariance(-1.0, 5.0)),
            (input = (m_out = NormalMeanPrecision(2.0, 2.0),            m_in1 = NormalMeanPrecision(-1.0, 3.0)),        output = NormalMeanVariance(-3.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_out = NormalMeanPrecision(-1.0, 2.0),           m_in1 = NormalMeanPrecision(-1.0, 3.0)),        output = NormalMeanVariance(0.0, (2.0 + 3.0) / (2.0*3.0))),
            (input = (m_out = NormalMeanPrecision(2.0, 2.0),            m_in1 = NormalMeanVariance(-1.0, 3.0)),         output = NormalMeanVariance(-3.0, 3.5)),
            (input = (m_out = NormalWeightedMeanPrecision(8.0, 4.0),    m_in1 = NormalMeanVariance(-3.0, 1.0)),         output = NormalMeanVariance(-5.0, 5/4)),
            (input = (m_out = NormalMeanVariance(-4.0, 2.0),            m_in1 = NormalWeightedMeanPrecision(6.0, 3.0)), output = NormalMeanVariance(6.0, 7/3)),
        ]

    end

    @testset "Belief Propagation: (m_out::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily)" begin

        @test_rules [ with_float_conversions = true ] typeof(-)(:in2, Marginalisation) [
            (input = (m_out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]),   m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),    output = MvNormalMeanCovariance([0.0, 0.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_out = MvNormalMeanCovariance([-1.0, 1.0], [3.0 2.0; 2.0 4.0]),  m_in1 = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),    output = MvNormalMeanCovariance([2.0, 2.0], [6.0 4.0; 4.0 8.0])),
            (input = (m_out = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]),    m_in1 = MvNormalMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0])),    output = MvNormalMeanCovariance([0.0, -7.0], [1.0 0.0; 0.0 4/3])),
            (input = (m_out = MvNormalMeanCovariance([1.0, -1.0], [3.0 1.0; 1.0 4.0]),  m_in1 = MvNormalMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0])),     output = MvNormalMeanCovariance([0.0, 5.0], [36/10 4/5; 4/5 44/10])),
            (input = (m_out = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]),    m_in1 = MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0])),    output = MvNormalMeanCovariance([-1/2, -7/3], [1.0 0.0; 0.0 4/3])),
            (input = (m_out = MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 4.0]),   m_in1 = MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0])),     output = MvNormalMeanCovariance([-6/5, 2/5], [36/10 4/5; 4/5 44/10])),
            (input = (m_out = MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]),   m_in1 = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0])),     output = MvNormalMeanCovariance([1/2, 7/3], [1.0 0.0; 0.0 4/3])),
            (input = (m_out = MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]),    m_in1 = MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 4.0])),    output = MvNormalMeanCovariance([6/5, -2/5], [36/10 4/5; 4/5 44/10])),
        ]

    end
end
end