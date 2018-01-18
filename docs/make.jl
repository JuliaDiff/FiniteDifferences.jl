using Documenter, FDM

makedocs(
    modules=[FDM],
    format=:html,
    pages=[
        "Home" => "index.md",
        "API" => "pages/api.md"
    ],
    sitename="FDM.jl",
    authors="Invenia Labs",
    assets=[
        "assets/invenia.css",
    ],
)

deploydocs(
    repo = "github.com/invenia/FDM.jl.git",
    julia = "0.6",
    target = "build",
    deps = nothing,
    make = nothing,
)
