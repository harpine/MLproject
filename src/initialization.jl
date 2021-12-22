using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
print("done")
include("./utilities.jl")