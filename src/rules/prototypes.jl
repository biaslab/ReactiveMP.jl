
include("uninformative/out.jl")

include("addition/marginals.jl")
include("addition/out.jl")
include("addition/in1.jl")
include("addition/in2.jl")

include("subtraction/marginals.jl")
include("subtraction/out.jl")
include("subtraction/in1.jl")
include("subtraction/in2.jl")

include("multiplication/marginals.jl")
include("multiplication/out.jl")
include("multiplication/in.jl")

include("gamma/out.jl")
include("gamma/marginals.jl")

include("gamma_shape_rate/out.jl")
include("gamma_shape_rate/marginals.jl")

include("beta/out.jl")
include("beta/marginals.jl")

include("dirichlet/marginals.jl")
include("dirichlet/out.jl")

include("matrix_dirichlet/out.jl")

include("categorical/out.jl")
include("categorical/p.jl")

include("bernoulli/p.jl")
include("bernoulli/out.jl")
include("bernoulli/marginals.jl")

include("gcv/x.jl")
include("gcv/y.jl")
include("gcv/z.jl")
include("gcv/k.jl")
include("gcv/w.jl")
include("gcv/marginals.jl")

include("kernel_gcv/x.jl")
include("kernel_gcv/y.jl")
include("kernel_gcv/z.jl")
include("kernel_gcv/marginals.jl")

include("mv_normal_mean_covariance/out.jl")
include("mv_normal_mean_covariance/mean.jl")
include("mv_normal_mean_covariance/marginals.jl")

include("normal_mean_variance/out.jl")
include("normal_mean_variance/mean.jl")
include("normal_mean_variance/var.jl")
include("normal_mean_variance/marginals.jl")

include("mv_normal_mean_precision/out.jl")
include("mv_normal_mean_precision/mean.jl")
include("mv_normal_mean_precision/precision.jl")
include("mv_normal_mean_precision/marginals.jl")

include("normal_mean_precision/out.jl")
include("normal_mean_precision/mean.jl")
include("normal_mean_precision/precision.jl")
include("normal_mean_precision/marginals.jl")

include("wishart/out.jl")
include("wishart/marginals.jl")

include("normal_mixture/switch.jl")
include("normal_mixture/m.jl")
include("normal_mixture/p.jl")
include("normal_mixture/out.jl")

include("gamma_mixture/switch.jl")
include("gamma_mixture/a.jl")
include("gamma_mixture/b.jl")
include("gamma_mixture/out.jl")

include("dot_product/marginals.jl")
include("dot_product/out.jl")
include("dot_product/in1.jl")
include("dot_product/in2.jl")

include("transition/marginals.jl")
include("transition/out.jl")
include("transition/in.jl")
include("transition/a.jl")

include("autoregressive/y.jl")
include("autoregressive/x.jl")
include("autoregressive/theta.jl")
include("autoregressive/gamma.jl")
include("autoregressive/marginals.jl")

include("probit/in.jl")
include("probit/out.jl")