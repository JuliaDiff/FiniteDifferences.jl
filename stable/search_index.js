var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#FDM.jl:-Finite-Difference-Methods-1",
    "page": "Home",
    "title": "FDM.jl: Finite Difference Methods",
    "category": "section",
    "text": "(Image: Build Status) (Image: codecov.io) (Image: Latest Docs)FDM.jl approximates derivatives of functions using finite difference methods."
},

{
    "location": "index.html#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": "Compute the first derivative of sin with a 5th order central method:julia> central_fdm(5, 1)(sin, 1) - cos(1)\n-1.247890679678676e-13Compute the second derivative of sin with a 5th order central method:julia> central_fdm(5, 2)(sin, 1) + sin(1)\n9.747314066999024e-12Construct a FDM on a custom grid:julia> method, report = fdm([-2, 0, 5], 1, report=true)\n(FDM.method, FDMReport:\n  order of method:       3\n  order of derivative:   1\n  grid:                  [-2, 0, 5]\n  coefficients:          [-0.357143, 0.3, 0.0571429]\n  roundoff error:        2.22e-16\n  bounds on derivatives: 1.00e+00\n  step size:             3.62e-06\n  accuracy:              6.57e-11\n)\n\njulia> method(sin, 1) - cos(1)\n-2.05648831297367e-11Compute a directional derivative:julia> f(x) = sum(x)\nf (generic function with 1 method)\n\njulia> central_fdm(5, 1)(ε -> f([1, 1, 1] + ε * [1, 2, 3]), 0) - 6\n-2.922107000813412e-13"
},

{
    "location": "pages/api.html#FDM.FDMReport",
    "page": "API",
    "title": "FDM.FDMReport",
    "category": "type",
    "text": "FDMReport\n\nDetails of a finite difference method to estimate a derivative. Instances of FDMReport Base.show nicely.\n\nFields\n\np::Int: Order of the method.\nq::Int: Order of the derivative that is estimated.\ngrid::Vector{<:Real}: Relative spacing of samples of f that are used by the method.\ncoefs::Vector{<:Real}: Weights of the samples of f.\nε::Real: Absolute roundoff error of the function evaluations.\nM::Real: Assumed upper bound of f and all its derivatives at x.\nĥ::Real: Step size.\nerr::Real: Estimated absolute accuracy.\n\n\n\n"
},

{
    "location": "pages/api.html#FDM.assert_approx_equal-NTuple{5,Any}",
    "page": "API",
    "title": "FDM.assert_approx_equal",
    "category": "method",
    "text": "assert_approx_equal(x, y, ε_abs, ε_rel[, desc])\n\nAssert that x is approximately equal to y.\n\nLet eps_z = eps_abs / eps_rel. Call x and y small if abs(x) < eps_z and abs(y) < eps_z, and call x and y large otherwise.  If this function returns True, then it is guaranteed that abs(x - y) < 2 eps_rel max(abs(x), abs(y)) if x and y are large, and abs(x - y) < 2 eps_abs if x and y are small.\n\nArguments\n\nx: First object to compare.\ny: Second object to compare.\nε_abs: Absolute tolerance.\nε_rel: Relative tolerance.\ndesc: Description of the comparison. Omit or set to false to have no description.\n\n\n\n"
},

{
    "location": "pages/api.html#FDM.backward_fdm-Tuple{Int64,Vararg{Any,N} where N}",
    "page": "API",
    "title": "FDM.backward_fdm",
    "category": "method",
    "text": "backward_fdm(p::Int, ...)\n\nConstruct a backward finite difference method of order p. See fdm for further details.\n\nArguments\n\np::Int: Order of the method.\n\nFurther takes, in the following order, the arguments q, ε, M, and report from fdm.\n\n\n\n"
},

{
    "location": "pages/api.html#FDM.central_fdm-Tuple{Int64,Vararg{Any,N} where N}",
    "page": "API",
    "title": "FDM.central_fdm",
    "category": "method",
    "text": "central_fdm(p::Int, ...)\n\nConstruct a central finite difference method of order p. See fdm for further details.\n\nArguments\n\np::Int: Order of the method.\n\nFurther takes, in the following order, the arguments q, ε, M, and report from fdm.\n\n\n\n"
},

{
    "location": "pages/api.html#FDM.fdm-Tuple{Array{#s102,1} where #s102<:Real,Int64}",
    "page": "API",
    "title": "FDM.fdm",
    "category": "method",
    "text": "function fdm(\n    grid::Vector{<:Real},\n    q::Int;\n    ε::Real=eps(),\n    M::Real=1,\n    report::Bool=false\n)\n\nConstruct a function method(f, x::Real, h::Real=ĥ) that takes in a function f, a point x in the domain of f, and optionally a step size h, and estimates the q\'th order derivative of f at x with a length(grid)\'th order finite difference method.\n\nArguments\n\ngrid::Vector{<:Real}: Relative spacing of samples of f that are used by the method.   The length of grid determines the order of the method.\nq::Int: Order of the derivative to estimate. q must be strictly less than the order   of the method.\n\nKeywords\n\nε::Real=eps(): Absolute roundoff error on the function evaluations.\nM::Real=1: Upper bound on f and all its derivatives.\nreport::Bool=false: Also return an instance of FDMReport containing information   about the method constructed.\n\n\n\n"
},

{
    "location": "pages/api.html#FDM.forward_fdm-Tuple{Int64,Vararg{Any,N} where N}",
    "page": "API",
    "title": "FDM.forward_fdm",
    "category": "method",
    "text": "forward_fdm(p::Int, ...)\n\nConstruct a forward finite difference method of order p. See fdm for further details.\n\nArguments\n\np::Int: Order of the method.\n\nFurther takes, in the following order, the arguments q, ε, M, and report from fdm.\n\n\n\n"
},

{
    "location": "pages/api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": "Modules = [FDM]\nPrivate = false"
},

]}
