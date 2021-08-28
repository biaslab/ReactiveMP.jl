module MvNormalMeanPrecisionTest

using Test
using ReactiveMP
using LinearAlgebra
using Distributions

@testset "MvNormalMeanPrecision" begin

    @testset "Constructor" begin
        @test MvNormalMeanPrecision <: AbstractMvNormal

        @test MvNormalMeanPrecision([ 1.0, 1.0 ]) == MvNormalMeanPrecision([ 1.0, 1.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanPrecision([ 1.0, 2.0 ]) == MvNormalMeanPrecision([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanPrecision([ 1, 2 ])     == MvNormalMeanPrecision([ 1.0, 2.0 ], [ 1.0, 1.0 ])
        @test MvNormalMeanPrecision([ 1f0, 2f0 ]) == MvNormalMeanPrecision([ 1f0, 2f0 ], [ 1f0, 1f0 ])

        @test eltype(MvNormalMeanPrecision([ 1.0, 1.0 ]))               === Float64
        @test eltype(MvNormalMeanPrecision([ 1.0, 1.0 ], [ 1.0, 1.0 ])) === Float64
        @test eltype(MvNormalMeanPrecision([ 1, 1 ]))                   === Float64
        @test eltype(MvNormalMeanPrecision([ 1, 1 ], [ 1, 1 ]))         === Float64
        @test eltype(MvNormalMeanPrecision([ 1f0, 1f0 ]))               === Float32
        @test eltype(MvNormalMeanPrecision([ 1f0, 1f0 ], [ 1f0, 1f0 ])) === Float32
    end

    @testset "distrname" begin
        @test ReactiveMP.distrname(MvNormalMeanPrecision(zeros(2))) === "MvNormalMeanPrecision"
    end

    @testset "Stats methods" begin
        
        μ    = [ 0.2, 3.0, 4.0 ]
        Λ    = [ 1.5 -0.3 0.1; -0.3 1.8 0.0; 0.1 0.0 3.5 ]
        dist = MvNormalMeanPrecision(μ, Λ)

        @test mean(dist)         == μ
        @test mode(dist)         == μ
        @test weightedmean(dist) == Λ * μ
        @test invcov(dist)       == Λ
        @test precision(dist)    == Λ
        @test cov(dist)          ≈ cholinv(Λ)
        @test std(dist)          ≈ cholsqrt(cholinv(Λ))
        @test all(mean_cov(dist) .≈ (μ, cholinv(Λ)))
        @test all(mean_invcov(dist) .≈ (μ, Λ))
        @test all(mean_precision(dist) .≈ (μ, Λ))
        @test all(weightedmean_cov(dist) .≈ (Λ * μ, cholinv(Λ)))
        @test all(weightedmean_invcov(dist) .≈ (Λ * μ, Λ))
        @test all(weightedmean_precision(dist) .≈ (Λ * μ, Λ))
        
        @test length(dist) == 3
        @test entropy(dist) ≈ 3.1517451983126357
        @test pdf(dist, [ 0.2, 3.0, 4.0 ])         ≈ 0.19171503573907536
        @test pdf(dist,  [0.202, 3.002, 4.002])    ≈ 0.19171258180232315
        @test logpdf(dist, [ 0.2, 3.0, 4.0 ])      ≈ -1.6517451983126357
        @test logpdf(dist,  [0.202, 3.002, 4.002]) ≈ -1.6517579983126356
        
    end

    @testset "Base methods" begin
        @test convert(MvNormalMeanPrecision{Float32}, MvNormalMeanPrecision([ 0.0, 0.0 ])) == MvNormalMeanPrecision([ 0f0, 0f0 ], [ 1f0, 1f0 ])
        @test convert(MvNormalMeanPrecision{Float64}, [ 0.0, 0.0 ], [ 2 0; 0 3 ]) == MvNormalMeanPrecision([ 0.0, 0.0 ], [ 2.0 0.0; 0.0 3.0 ])

        @test length(MvNormalMeanPrecision([ 0.0, 0.0 ]))      === 2
        @test length(MvNormalMeanPrecision([ 0.0, 0.0, 0.0 ])) === 3
        @test ndims(MvNormalMeanPrecision([ 0.0, 0.0 ]))       === 2
        @test ndims(MvNormalMeanPrecision([ 0.0, 0.0, 0.0 ]))  === 3
        @test size(MvNormalMeanPrecision([ 0.0, 0.0 ]))        === (2, )
        @test size(MvNormalMeanPrecision([ 0.0, 0.0, 0.0 ]))   === (3, )
    end

    @testset "vague" begin
        @test_throws MethodError vague(MvNormalMeanPrecision)

        d1 = vague(MvNormalMeanPrecision, 2)

        @test typeof(d1) <: MvNormalMeanPrecision
        @test mean(d1)   == zeros(2)
        @test invcov(d1) == Matrix(Diagonal(ReactiveMP.tiny * ones(2)))
        @test ndims(d1) == 2

        d2 = vague(MvNormalMeanPrecision, 3)

        @test typeof(d2) <: MvNormalMeanPrecision
        @test mean(d2)   == zeros(3)
        @test invcov(d2) == Matrix(Diagonal(ReactiveMP.tiny * ones(3)))
        @test ndims(d2)  == 3
    end

    @testset "prod" begin
        
        @test prod(ProdAnalytical(), MvNormalMeanPrecision([ -1, -1 ], [ 2, 2 ]), MvNormalMeanPrecision([ 1, 1 ], [ 2, 4 ])) ≈ MvNormalWeightedMeanPrecision([ 0, 2 ], [ 4, 6 ])

        μ    = [ 1.0, 2.0, 3.0 ]
        Λ    = diagm(1 ./ [1.0, 2.0, 3.0])
        dist = MvNormalMeanPrecision(μ, Λ)

        @test prod(ProdAnalytical(), dist, dist) ≈ MvNormalWeightedMeanPrecision([2.0, 2.0, 2.0], diagm([2.0, 1.0, 2/3]))

    end

    @testset "Primitive types conversion" begin
        @test convert(MvNormalMeanPrecision, zeros(2), Matrix(Diagonal(ones(2)))) == MvNormalMeanPrecision(zeros(2), Matrix(Diagonal(ones(2))))
        @test begin
            m = rand(5)
            c = Matrix(Symmetric(rand(5, 5)))
            convert(MvNormalMeanPrecision, m, c) == MvNormalMeanPrecision(m, c)
        end
    end

end

end