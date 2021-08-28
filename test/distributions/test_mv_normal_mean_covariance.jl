module MvNormalMeanCovarianceTest

using Test
using ReactiveMP
using LinearAlgebra
using Distributions

@testset "MvNormalMeanCovariance" begin

    @testset "Constructor" begin
        @test MvNormalMeanCovariance <: AbstractMvNormal

        @test MvNormalMeanCovariance([ 1.0, 1.0 ]) == MvNormalMeanCovariance([ 1.0, 1.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanCovariance([ 1.0, 2.0 ]) == MvNormalMeanCovariance([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanCovariance([ 1, 2 ])     == MvNormalMeanCovariance([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanCovariance([ 1f0, 2f0 ]) == MvNormalMeanCovariance([ 1f0, 2f0 ], [ 1f0, 1f0 ])

        @test eltype(MvNormalMeanCovariance([ 1.0, 1.0 ]))               === Float64
        @test eltype(MvNormalMeanCovariance([ 1.0, 1.0 ], [ 1.0, 1.0 ])) === Float64
        @test eltype(MvNormalMeanCovariance([ 1, 1 ]))                   === Float64
        @test eltype(MvNormalMeanCovariance([ 1, 1 ], [ 1, 1 ]))         === Float64
        @test eltype(MvNormalMeanCovariance([ 1f0, 1f0 ]))               === Float32
        @test eltype(MvNormalMeanCovariance([ 1f0, 1f0 ], [ 1f0, 1f0 ])) === Float32
    end

    @testset "distrname" begin
        @test ReactiveMP.distrname(MvNormalMeanCovariance(zeros(2))) === "MvNormalMeanCovariance"
    end

    @testset "Stats methods" begin
        
        μ    = [ 0.2, 3.0, 4.0 ]
        Σ    = [ 1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalMeanCovariance(μ, Σ)

        @test mean(dist)         == μ
        @test mode(dist)         == μ
        @test weightedmean(dist) == cholinv(Σ) * μ
        @test invcov(dist)       ≈  cholinv(Σ)
        @test precision(dist)    ≈  cholinv(Σ)
        @test cov(dist)          == Σ
        @test std(dist)          ≈ cholsqrt(Σ)
        @test all(mean_cov(dist) .≈ (μ, Σ))
        @test all(mean_invcov(dist) .≈ (μ, cholinv(Σ)))
        @test all(mean_precision(dist) .≈ (μ, cholinv(Σ)))
        @test all(weightedmean_cov(dist) .≈ (cholinv(Σ) * μ, Σ))
        @test all(weightedmean_invcov(dist) .≈ (cholinv(Σ) * μ, cholinv(Σ)))
        @test all(weightedmean_precision(dist) .≈ (cholinv(Σ) * μ, cholinv(Σ)))
        
        @test length(dist) == 3
        @test entropy(dist) ≈ 5.361886000915401
        @test pdf(dist, [ 0.2, 3.0, 4.0 ])         ≈ 0.021028302702542
        @test pdf(dist,  [0.202, 3.002, 4.002])    ≈ 0.021028229679079503
        @test logpdf(dist, [ 0.2, 3.0, 4.0 ])      ≈ -3.8618860009154012
        @test logpdf(dist,  [0.202, 3.002, 4.002]) ≈ -3.861889473548943
        
    end

    @testset "Base methods" begin
        @test convert(MvNormalMeanCovariance{Float32}, MvNormalMeanCovariance([ 0.0, 0.0 ])) == MvNormalMeanCovariance([ 0f0, 0f0 ], [ 1f0, 1f0 ])
        @test convert(MvNormalMeanCovariance{Float64}, [ 0.0, 0.0 ], [ 2 0; 0 3 ]) == MvNormalMeanCovariance([ 0.0, 0.0 ], [ 2.0 0.0; 0.0 3.0 ])

        @test length(MvNormalMeanCovariance([ 0.0, 0.0 ]))      === 2
        @test length(MvNormalMeanCovariance([ 0.0, 0.0, 0.0 ])) === 3
        @test ndims(MvNormalMeanCovariance([ 0.0, 0.0 ]))       === 2
        @test ndims(MvNormalMeanCovariance([ 0.0, 0.0, 0.0 ]))  === 3
        @test size(MvNormalMeanCovariance([ 0.0, 0.0 ]))        === (2, )
        @test size(MvNormalMeanCovariance([ 0.0, 0.0, 0.0 ]))   === (3, )
    end

    @testset "vague" begin
        @test_throws MethodError vague(MvNormalMeanCovariance)

        d1 = vague(MvNormalMeanCovariance, 2)

        @test typeof(d1) <: MvNormalMeanCovariance
        @test mean(d1)  == zeros(2)
        @test cov(d1)   == Matrix(Diagonal(ReactiveMP.huge * ones(2)))
        @test ndims(d1) == 2

        d2 = vague(MvNormalMeanCovariance, 3)

        @test typeof(d2) <: MvNormalMeanCovariance
        @test mean(d2)  == zeros(3)
        @test cov(d2)   == Matrix(Diagonal(ReactiveMP.huge * ones(3)))
        @test ndims(d2) == 3
    end

    @testset "prod" begin
        
        @test prod(ProdAnalytical(), MvNormalMeanCovariance([ -1, -1 ], [ 2, 2 ]), MvNormalMeanCovariance([ 1, 1 ], [ 2, 4 ])) ≈ MvNormalWeightedMeanPrecision([ 0, -1/4 ], [ 1, 3/4])

        μ    = [ 1.0, 2.0, 3.0 ]
        Σ    = diagm([1.0, 2.0, 3.0])
        dist = MvNormalMeanCovariance(μ, Σ)

        @test prod(ProdAnalytical(), dist, dist) ≈ MvNormalWeightedMeanPrecision([2.0, 2.0, 2.0], diagm([2.0, 1.0, 2/3]))

    end

    @testset "Primitive types conversion" begin
        @test convert(MvNormalMeanCovariance, zeros(2), Matrix(Diagonal(ones(2)))) == MvNormalMeanCovariance(zeros(2), Matrix(Diagonal(ones(2))))
        @test begin
            m = rand(5)
            c = Matrix(Symmetric(rand(5, 5)))
            convert(MvNormalMeanCovariance, m, c) == MvNormalMeanCovariance(m, c)
        end
    end

end

end