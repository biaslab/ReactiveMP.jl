@testitem "DivisionOf" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase

    # `DivisionOf` is internal to the extension
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    d1 = NormalMeanVariance(0, 1)
    d2 = NormalMeanVariance(1, 2)

    @test d1 ≈ prod(GenericProd(), ext.DivisionOf(d1, d2), d2)
    @test d1 ≈ prod(GenericProd(), d2, ext.DivisionOf(d1, d2))
    @test ext.DivisionOf(d1, d2) == prod(GenericProd(), ext.DivisionOf(d1, d2), missing)
    @test ext.DivisionOf(d1, d2) == prod(GenericProd(), missing, ext.DivisionOf(d1, d2))
end

@testitem "create_project_to_ins type stability" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, Test
    using ReactiveMP: CVIProjection
    using JET

    # `create_project_to_ins` is internal to the extension
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    @testset "Complete type stability tests for create_project_to_ins" begin
        # Test Case 1: Default form (nothing)
        let
            method = CVIProjection()
            m_in = NormalMeanVariance(0.0, 1.0)
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:NormalMeanVariance}
        end

        # Test Case 2: Custom form specified
        let
            form = ProjectedTo(MvNormalMeanScalePrecision, 2)
            method = CVIProjection(in_prjparams = (in_1 = form,))
            m_in = NormalMeanVariance(0.0, 1.0)  # Input type different from target
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:MvNormalMeanScalePrecision}
        end

        # Test Case 3: Multiple forms specified
        let
            forms = (in_1 = ProjectedTo(NormalMeanVariance), in_2 = ProjectedTo(MvNormalMeanCovariance))
            method = CVIProjection(in_prjparams = forms)
            m_in = Gamma(2.0, 2.0)

            for k in 1:2
                @test_opt ext.create_project_to_ins(method, m_in, k)
                result = ext.create_project_to_ins(method, m_in, k)

                if k == 1
                    @test result isa ProjectedTo{<:NormalMeanVariance}
                else
                    @test result isa ProjectedTo{<:MvNormalMeanCovariance}
                end
            end
        end

        # Test Case 4: not form but just a gradient descent parameters
        let
            params = ExponentialFamilyProjection.DefaultProjectionParameters()
            method = CVIProjection(in_prjparams = (in_1 = params,))
            m_in = Gamma(1.0, 1.0)
            k = 1

            @test_opt ext.create_project_to_ins(method, m_in, k)
            result = ext.create_project_to_ins(method, m_in, k)
            @test result isa ProjectedTo{<:Gamma}
        end
    end
end

@testitem "create_project_to type stability" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, Test
    using ReactiveMP: CVIProjection
    using JET

    # `create_project_to_ins` is internal to the extension
    ext = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ext)
    using .ext

    @testset "Complete type stability tests for create_project_to" begin
        # Test Case 1: Default form (Nothing case)
        let
            method = CVIProjection()
            q_out = NormalMeanVariance(0.0, 1.0)
            q_out_samples = [[1.0], [2.0], [3.0]]

            @test_opt ext.create_project_to(method, q_out, q_out_samples)
            result = ext.create_project_to(method, q_out, q_out_samples)

            @test result isa ProjectedTo{<:NormalMeanVariance}
            @test ExponentialFamilyProjection.get_projected_to_dims(result) == size(first(q_out_samples))
        end

        # Test Case 2: Existing ProjectedTo form
        let
            form = ProjectedTo(MvNormalMeanScalePrecision, 2)
            method = CVIProjection(out_prjparams = form)
            q_out = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])
            q_out_samples = [[1.0, 2.0], [3.0, 4.0]]

            @test_opt ext.create_project_to(method, q_out, q_out_samples)
            result = ext.create_project_to(method, q_out, q_out_samples)

            @test result === method.out_prjparams
            @test result isa ProjectedTo{<:MvNormalMeanScalePrecision}
        end

        # Test Case 3: Custom ProjectionParameters
        let
            params = ExponentialFamilyProjection.DefaultProjectionParameters()
            method = CVIProjection(out_prjparams = params)
            q_out = Gamma(2.0, 2.0)
            q_out_samples = [[1.0], [2.0], [3.0]]

            @test_opt ext.create_project_to(method, q_out, q_out_samples)
            result = ext.create_project_to(method, q_out, q_out_samples)

            @test result isa ProjectedTo{<:Gamma}
            @test result.parameters === method.out_prjparams
            @test ExponentialFamilyProjection.get_projected_to_dims(result) == size(first(q_out_samples))
        end

        # Test Case 4: Different dimensions and distributions
        let
            method = CVIProjection()
            distributions_and_samples = [
                (MvNormalMeanScalePrecision([1, 2, 3], 1), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), (NormalMeanVariance(0.0, 1.0), [[1.0], [2.0]]), (Gamma(2.0, 2.0), [[1.0], [2.0]])
            ]

            for (dist, samples) in distributions_and_samples
                @test_opt ext.create_project_to(method, dist, samples)
                result = ext.create_project_to(method, dist, samples)

                @test result isa ProjectedTo
                @test ExponentialFamilyProjection.get_projected_to_dims(result) == size(first(samples))
                @test result.parameters isa ExponentialFamilyProjection.ProjectionParameters
            end
        end
    end
end
