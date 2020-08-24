using Documenter, FiniteDifferences

makedocs(
    modules=[FiniteDifferences],
    format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"),
    pages=["Home" => "index.md", "API" => "pages/api.md"],
    authors="Invenia Labs",
    checkdocs=:exports,
    repo="https://github.com/JuliaDiff/FiniteDifferences.jl/blob/{commit}{path}#L{line}",
    sitename="FiniteDifferences.jl",
    strict=true,
)

deploydocs(repo="github.com/JuliaDiff/FiniteDifferences.jl.git")
