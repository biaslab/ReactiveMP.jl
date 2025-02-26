@testitem "Basic checks for marginal rule" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase
    import ReactiveMP: @test_rules, @test_marginalrules

    @testset "f(x) -> x, x~EF, out~EF" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        # Since we use `identity` as a function we expect that the result of marginal computation is a product of `m_out` and `m_in`
        inputs = [
            (NormalMeanVariance(0, 1), NormalMeanVariance(0, 1)),
            (Gamma(2, 2), Gamma(2, 2)),
            (Beta(1, 1), Beta(1, 1)),
            (MvNormalMeanCovariance([0.5, 0.5]), MvNormalMeanCovariance([0.5, 0.5])),
            (MvNormalMeanCovariance([0.5, 0.5, -1.0]), MvNormalMeanCovariance([0.5, 2.5, -3.0]))
        ]
        for input in inputs
            m_in = input[1]
            m_out = input[1]
            q_factorised = @call_marginalrule DeltaFn{identity}(:ins) (m_out = m_out, m_ins = ManyOf(m_in), meta = meta)
            @test length(q_factorised) === 1
            q_in_1 = component(q_factorised, 1)
            @test q_in_1 ≈ prod(GenericProd(), m_in, m_out) atol = 1e-1
        end
    end

    @testset "f(x, y) -> [x, y], x~Normal, y~Normal, out~MvNormal (marginalization)" begin
        f(x, y) = [x, y]
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{f}(:ins) [(
            input = (m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]), m_ins = ManyOf(NormalMeanVariance(0, 1), NormalMeanVariance(1, 2)), meta = meta),
            output = FactorizedJoint((NormalMeanVariance(1 / 3, 2 / 3), NormalMeanVariance(1.0, 1.0)))
        )]
    end

    @testset "f(x) -> x, x~EF, out~EF with Binomial" begin
        meta = DeltaMeta(method = CVIProjection(), inverse = nothing)
        inputs_outputs = [(Binomial(3, 0.9), Binomial(7, 0.4)), (Binomial(8, 0.9), Binomial(8, 0.9)), (Binomial(5, 0.01), Binomial(6, 0.98))]
        for input_output in inputs_outputs
            m_in = first(input_output)
            m_out = last(input_output)
            q_factorised = @call_marginalrule DeltaFn{identity}(:ins) (m_out = m_out, m_ins = ManyOf(m_in), meta = meta)
            @test length(q_factorised) === 1
            component1 = component(q_factorised, 1)
            q_prod = prod(PreserveTypeProd(ExponentialFamilyDistribution), m_in, m_out)
            grid = getsupport(q_prod)
            mean_prod = sum(grid .* pdf(q_prod, grid))
            @test mean(component1) ≈ mean_prod atol = 1e-1
        end
    end

    @testset "f(x) -> x, x~EF, out~EF with Categorical" begin
        meta = DeltaMeta(
            method = CVIProjection(
                in_prjparams = (in_1 = ExponentialFamilyProjection.ProjectionParameters(strategy = ExponentialFamilyProjection.ControlVariateStrategy(nsamples = 4000)),)
            )
        )
        inputs_outputs = [
            (Categorical([1 / 4, 1 / 4, 1 / 2]), Categorical([1 / 2, 1 / 8, 3 / 8])),
            (Categorical([1 / 8, 1 / 8, 3 / 4]), Categorical([1 / 16, 13 / 16, 1 / 8])),
            (Categorical([1 / 7, 1 / 7, 2 / 7, 3 / 7]), Categorical([1 / 8, 2 / 8, 2 / 8, 3 / 8]))
        ]
        for input_output in inputs_outputs
            m_in = first(input_output)
            m_out = last(input_output)
            q_factorised = @call_marginalrule DeltaFn{identity}(:ins) (m_out = m_out, m_ins = ManyOf(m_in), meta = meta)
            @test length(q_factorised) === 1
            component1 = component(q_factorised, 1)
            q_prod = prod(GenericProd(), m_in, m_out)
            @test mean(component1) ≈ mean(q_prod) atol = 1e-1
        end
    end
end

@testitem "CVIProjection form access tests" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra
    import ReactiveMP: get_kth_in_form

    @testset "Testing input edge form access with get_kth_in_form" begin
        # Create forms for specific inputs
        form1 = ProjectedTo(NormalMeanVariance)

        form2 = ProjectedTo(MvNormalMeanScalePrecision, 2)

        # Check form access behavior
        method_with_forms = CVIProjection(in_prjparams = (in_1 = form1, in_2 = form2))
        @test !isnothing(get_kth_in_form(method_with_forms, 1))
        @test !isnothing(get_kth_in_form(method_with_forms, 2))
        @test isnothing(get_kth_in_form(method_with_forms, 3))  # Non-existent index

        method_default = CVIProjection()
        @test isnothing(get_kth_in_form(method_default, 1))
        @test isnothing(get_kth_in_form(method_default, 2))

        # Test with partial specification
        meta_partial = DeltaMeta(method = CVIProjection(
            in_prjparams = (in_2 = form2,), # Only specify second input
            sampling_strategy = FullSampling(10)
        ), inverse = nothing)

        # Setup messages
        m_out = MvNormalMeanCovariance([2.0, 3.0], Matrix{Float64}(I, 2, 2))
        m_in1 = Gamma(2.0, 2.0)
        m_in2 = MvNormalMeanCovariance([1.0, 1.0], [2.0 0.0; 0.0 2.0])

        f(x, y) = x .* y

        result = @call_marginalrule DeltaFn{f}(:ins) (m_out = m_out, m_ins = ManyOf(m_in1, m_in2), meta = meta_partial)

        # First input should use default form (nothing specified)
        # Second input should be MvNormalMeanScalePrecision as specified
        @test isa(result[1], Gamma)
        @test isa(result[2], MvNormalMeanScalePrecision)
    end
end

@testitem "CVIProjection proposal distribution convergence tests" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase, LinearAlgebra
    using Random, Distributions

    @testset "Posterior approximation quality" begin
        rng = MersenneTwister(123)
        method = CVIProjection(rng = rng, sampling_strategy = FullSampling(2000))
        meta = DeltaMeta(method = method, inverse = nothing)

        f(x, y) = x * y

        # Define distributions
        m_out = NormalMeanVariance(2.0, 0.1)
        m_in1 = NormalMeanVariance(0.0, 2.0)
        m_in2 = NormalMeanVariance(0.0, 2.0)

        # Function to compute unnormalized log posterior for a sample
        function log_posterior(x, y)
            return logpdf(m_in1, x) + logpdf(m_in2, y) + logpdf(m_out, f(x, y))
        end

        # Estimate KL divergence using samples
        function estimate_kl_divergence(q_result)
            n_samples = 10000
            samples_q = [(rand(rng, q_result[1]), rand(rng, q_result[2])) for _ in 1:n_samples]

            # Compute E_q[log q(x,y) - log p(x,y)]
            log_q_terms = [logpdf(q_result[1], x) + logpdf(q_result[2], y) for (x, y) in samples_q]
            log_p_terms = [log_posterior(x, y) for (x, y) in samples_q]

            return mean(log_q_terms .- log_p_terms)
        end

        # Run multiple iterations and collect KL divergences
        n_iterations = 10
        kl_divergences = Vector{Float64}(undef, n_iterations)

        for i in 1:n_iterations
            result = @call_marginalrule DeltaFn{f}(:ins) (m_out = m_out, m_ins = ManyOf(m_in1, m_in2), meta = meta)
            kl_divergences[i] = estimate_kl_divergence(result)
        end

        @test kl_divergences[1] > kl_divergences[end]
    end
end

@testitem "Basic checks for marginal rule with mean based approximation" begin
    using ExponentialFamily, ExponentialFamilyProjection, BayesBase
    import ReactiveMP: @test_rules, @test_marginalrules

    @testset "f(x, y) -> [x, y], x~Normal, y~Normal, out~MvNormal (marginalization)" begin
        f(x, y) = [x, y]
        meta = DeltaMeta(method = CVIProjection(sampling_strategy = MeanBased()), inverse = nothing)
        @test_marginalrules [check_type_promotion = false, atol = 1e-1] DeltaFn{f}(:ins) [(
            input = (m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2]), m_ins = ManyOf(NormalMeanVariance(0, 1), NormalMeanVariance(1, 2)), meta = meta),
            output = FactorizedJoint((NormalMeanVariance(1 / 3, 2 / 3), NormalMeanVariance(1.0, 1.0)))
        )]
    end
end

@testitem "DeltaNode - CVI sampling strategy performance comparison" begin
    using Test
    using BenchmarkTools
    using BayesBase, ExponentialFamily, ExponentialFamilyProjection

    f(x, y) = [x, y]

    function run_marginal_test(strategy)
        meta = DeltaMeta(method = CVIProjection(sampling_strategy = strategy))
        m_out = MvGaussianMeanCovariance(ones(2), [2 0; 0 2])
        m_in1 = NormalMeanVariance(0.0, 2.0)
        m_in2 = NormalMeanVariance(0.0, 2.0)
        return @belapsed begin
            @call_marginalrule DeltaFn{f}(:ins) (m_out = $m_out, m_ins = ManyOf($m_in1, $m_in2), meta = $meta)
        end samples = 2
    end

    # Run benchmarks
    full_time = run_marginal_test(FullSampling(10))
    mean_time = run_marginal_test(MeanBased())

    @test mean_time < full_time

    # Optional: Print the actual times for verification
    @info "Sampling strategy performance" full_time mean_time ratio = (full_time / mean_time)
end
