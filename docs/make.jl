using ObjectiveMapping
using Documenter

DocMeta.setdocmeta!(ObjectiveMapping, :DocTestSetup, :(using ObjectiveMapping); recursive=true)

makedocs(;
    modules=[ObjectiveMapping],
    authors="A Meza",
    sitename="ObjectiveMapping.jl",
    format=Documenter.HTML(;
        canonical="https://anthony-meza.github.io/ObjectiveMapping.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/anthony-meza/ObjectiveMapping.jl",
    devbranch="main",
)
