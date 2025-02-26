@testitem "CVI Projection Extension Tests" begin
    using ExponentialFamily
    using ExponentialFamilyProjection
    using BayesBase
    using ReactiveMP
    using Distributions
    using Random

    ReactiveMPProjectionExt = Base.get_extension(ReactiveMP, :ReactiveMPProjectionExt)
    @test !isnothing(ReactiveMPProjectionExt)
    using .ReactiveMPProjectionExt

    @testset "create_density_function" begin
        # Mock functions and data for testing
        pre_samples = [1.0, 2.0, 3.0]

        # Mock message with a simple normal distribution
        m_in = NormalMeanVariance(0.0, 1.0)

        # Mock logp_nc_drop_index function that just returns a constant + the input value
        logp_nc_drop_index = (z, i, samples) -> -0.5 * z^2

        # Test when forms match (should not include the message logpdf)
        forms_match = true
        df_match = ReactiveMPProjectionExt.create_density_function(forms_match, 1, pre_samples, logp_nc_drop_index, m_in)
        @test df_match(0.5) ≈ logp_nc_drop_index(0.5, 1, pre_samples)
        @test df_match(1.0) ≈ -0.5 # Just the logp_nc_drop_index result

        # Test when forms don't match (should include the message logpdf)
        forms_match = false
        df_no_match = ReactiveMPProjectionExt.create_density_function(forms_match, 1, pre_samples, logp_nc_drop_index, m_in)
        # Expected: logp_nc_drop_index + logpdf of the message
        expected_value = logp_nc_drop_index(0.5, 1, pre_samples) + logpdf(m_in, 0.5)
        @test df_no_match(0.5) ≈ expected_value
    end

    @testset "optimize_parameters" begin
        # Test with normal distribution - we can derive exact expected results
        m_in = NormalMeanVariance(0.0, 1.0)  # Prior: mean=0, variance=1 (precision=1)
        m_ins = [m_in]
        pre_samples = [0.0, 0.5, -0.5]
        method = CVIProjection()

        # Case 1: Quadratic log-likelihood centered at 0 (-0.5*z²) corresponds to Normal(0,1)
        # When combining Normal(0,1) prior with Normal(0,1) likelihood:
        # Expected posterior: Normal(0, 0.5) - precision adds (1+1=2, variance=1/2=0.5)
        log_fn1 = (z, i, samples) -> -0.5 * z^2
        result1 = ReactiveMPProjectionExt.optimize_parameters(1, pre_samples, m_ins, log_fn1, method)

        @test result1 isa NormalMeanVariance
        @test mean(result1) ≈ 0.0 atol = 1e-1
        @test var(result1) ≈ 0.5 atol = 1e-1

        # Case 2: Quadratic centered at 2.0 (-0.5*(z-2)²) corresponds to Normal(2,1)
        # Combining Normal(0,1) prior with Normal(2,1) likelihood:
        # Expected posterior: Normal(1, 0.5) - weighted average of means
        log_fn2 = (z, i, samples) -> -0.5 * (z - 2.0)^2
        result2 = ReactiveMPProjectionExt.optimize_parameters(1, pre_samples, m_ins, log_fn2, method)

        @test result2 isa NormalMeanVariance
        @test mean(result2) ≈ 1.0 atol = 1e-1  # (0*1 + 2*1)/(1+1) = 1.0
        @test var(result2) ≈ 0.5 atol = 1e-1  # 1/(1+1) = 0.5

        # Case 3: Stronger quadratic (-2.0*(z-2)²) corresponds to Normal(2,0.25)
        # Combining Normal(0,1) prior with Normal(2,0.25) likelihood:
        # Expected posterior: Normal(1.6, 0.2) 
        log_fn3 = (z, i, samples) -> -2.0 * (z - 2.0)^2
        result3 = ReactiveMPProjectionExt.optimize_parameters(1, pre_samples, m_ins, log_fn3, method)

        @test result3 isa NormalMeanVariance
        @test mean(result3) ≈ 1.6 atol = 1e-1 # (0*1 + 2*4)/(1+4) = 8/5 = 1.6
        @test var(result3) ≈ 0.2 atol = 1e-1 # 1/(1+4) = 0.2

        # Case 4: Test with a different prior
        m_in2 = NormalMeanVariance(1.0, 2.0)  # Prior: mean=1, variance=2 (precision=0.5)
        m_ins2 = [m_in2]

        # Combining Normal(1,2) prior with Normal(2,1) likelihood:
        # Expected posterior: Normal(5/3, 2/3)
        result4 = ReactiveMPProjectionExt.optimize_parameters(1, pre_samples, m_ins2, log_fn2, method)

        @test result4 isa NormalMeanVariance
        @test mean(result4) ≈ 5 / 3 atol = 1e-1 # (1*0.5 + 2*1)/(0.5+1) = 1.67
        @test var(result4) ≈ 2 / 3 atol = 1e-1 # 1/(0.5+1) = 0.67
    end
end
