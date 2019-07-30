using Documenter, FiniteDifferences

makedocs(
    modules=[FiniteDifferences],
    format=:html,
    pages=[
        "Home" => "index.md",
        "API" => "pages/api.md"
    ],
    sitename="FiniteDifferences.jl",
    authors="Invenia Labs",
    assets=[
        "assets/invenia.css",
    ],
)

deploydocs(
    repo = "github.com/JuliaDiff/FiniteDifferences.jl.git",
    julia = "1.0",
    target = "build",
    deps = nothing,
    make = nothing,
)
