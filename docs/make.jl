using Documenter, ReactiveMP

## https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988
## https://gr-framework.org/workstations.html#no-output
ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(ReactiveMP, :DocTestSetup, :(using ReactiveMP, Distributions); recursive=true)

makedocs(
    modules  = [ ReactiveMP ],
    strict   = [ :doctest, :eval_block, :example_block, :meta_block, :parse_error, :setup_block ],
    clean    = true,
    sitename = "ReactiveMP.jl",
    pages    = [
        "Introduction"    => "index.md",
        "Custom functionality" => [
            "Custom functional form" => "custom/custom-functional-form.md",
        ],
        "Library" => [
            "Messages"     => "lib/message.md",
            "Factor nodes" => [ 
                "Overview" => "lib/nodes/nodes.md",
                "Flow"     => "lib/nodes/flow.md"
            ],
            "Prod implementation" => "lib/prod.md",
            "Helper utils"        => "lib/helpers.md",
            "Algebra utils"       => [
                "Common" => "lib/algebra/common.md"
            ],
            "Exported methods"    => "lib/methods.md"
        ],
        "Contributing" => "extra/contributing.md",
    ],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/biaslab/ReactiveMP.jl.git"
    )
end
