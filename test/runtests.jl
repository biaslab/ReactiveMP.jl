module ReactiveMPTest

using Test, Documenter, ReactiveMP
using TestSetExtensions

using Aqua
Aqua.test_all(ReactiveMP; ambiguities=false)

include("test_helpers.jl")

using .ReactiveMPTestingHelpers

# doctest(ReactiveMP)

@testset ExtendedTestSet "ReactiveMP" begin

    enabled_tests = lowercase.(ARGS)

    function key_to_filename(key)
        splitted = split(key, ":")
        return length(splitted) === 1 ? string("test_", first(splitted), ".jl") : string(join(splitted[1:end - 1], "/"), "/test_", splitted[end], ".jl")
    end

    function filename_to_key(filename)
        splitted   = split(filename, "/")
        if length(splitted) === 1
            return replace(replace(first(splitted), ".jl" => ""), "test_" => "")
        else
            path, name = splitted[1:end - 1], splitted[end]
            return string(join(path, ":"), ":", replace(replace(name, ".jl" => ""), "test_" => ""))
        end
    end

    function addtests(filename)
        key = filename_to_key(filename)
        if isempty(enabled_tests) || key in enabled_tests
            include(filename)
        end
    end

    @testset "Testset helpers" begin
        @test key_to_filename(filename_to_key("distributions/test_normal_mean_variance.jl")) == "distributions/test_normal_mean_variance.jl"
        @test filename_to_key(key_to_filename("distributions:normal_mean_variance")) == "distributions:normal_mean_variance"
        @test key_to_filename(filename_to_key("test_message.jl")) == "test_message.jl"
        @test filename_to_key(key_to_filename("message")) == "message"
    end

    addtests("algebra/test_correction.jl")

    addtests("test_math.jl")

    addtests("test_distributions.jl")
    addtests("distributions/test_common.jl")
    addtests("distributions/test_bernoulli.jl")
    addtests("distributions/test_beta.jl")
    addtests("distributions/test_categorical.jl")
    addtests("distributions/test_contingency.jl")
    addtests("distributions/test_dirichlet_matrix.jl")
    addtests("distributions/test_dirichlet.jl")
    addtests("distributions/test_gamma.jl")
    addtests("distributions/test_mv_normal_mean_covariance.jl")
    addtests("distributions/test_mv_normal_mean_precision.jl")
    addtests("distributions/test_mv_normal_weighted_mean_precision.jl")
    addtests("distributions/test_normal_mean_variance.jl")
    addtests("distributions/test_normal_mean_precision.jl")
    addtests("distributions/test_normal_weighted_mean_precision.jl")
    addtests("distributions/test_normal.jl")
    addtests("distributions/test_pointmass.jl")
    addtests("distributions/test_wishart.jl")

    addtests("test_message.jl")
    
    addtests("test_variable.jl")

    addtests("test_node.jl")
    addtests("nodes/test_addition.jl")
    addtests("nodes/test_bifm.jl")
    addtests("nodes/test_bifm_helper.jl")
    addtests("nodes/test_probit.jl")

    addtests("rules/addition/test_in1.jl")
    addtests("rules/addition/test_in2.jl")
    addtests("rules/addition/test_out.jl")

    addtests("rules/bifm/test_marginals.jl")
    addtests("rules/bifm/test_in.jl")
    addtests("rules/bifm/test_out.jl")
    addtests("rules/bifm/test_zprev.jl")
    addtests("rules/bifm/test_znext.jl")

    addtests("rules/bifm_helper/test_in.jl")
    addtests("rules/bifm_helper/test_out.jl")

    addtests("rules/normal_mean_variance/test_out.jl")
    addtests("rules/normal_mean_variance/test_mean.jl")
    addtests("rules/normal_mean_variance/test_var.jl")

    addtests("rules/normal_mean_precision/test_out.jl")
    addtests("rules/normal_mean_precision/test_mean.jl")
    addtests("rules/normal_mean_precision/test_precision.jl")

    addtests("rules/mv_normal_mean_covariance/test_out.jl")
    addtests("rules/mv_normal_mean_covariance/test_mean.jl")

    addtests("rules/mv_normal_mean_precision/test_out.jl")
    addtests("rules/mv_normal_mean_precision/test_mean.jl")
    addtests("rules/mv_normal_mean_precision/test_precision.jl")  

    addtests("rules/probit/test_out.jl")
    addtests("rules/probit/test_in.jl")

    addtests("rules/wishart/test_marginals.jl")
    addtests("rules/wishart/test_out.jl")

end

end
