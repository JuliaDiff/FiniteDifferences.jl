var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#FiniteDifferences.jl:-Finite-Difference-Methods-1",
    "page": "Home",
    "title": "FiniteDifferences.jl: Finite Difference Methods",
    "category": "section",
    "text": "(Image: Build Status) (Image: codecov.io) (Image: Latest Docs)FiniteDifferences.jl approximates derivatives of functions using finite difference methods."
},

{
    "location": "index.html#Examples-1",
    "page": "Home",
    "title": "Examples",
    "category": "section",
    "text": "Compute the first derivative of sin with a 5th order central method:julia> central_fdm(5, 1)(sin, 1) - cos(1)\n-1.247890679678676e-13Compute the second derivative of sin with a 5th order central method:julia> central_fdm(5, 2)(sin, 1) + sin(1)\n9.747314066999024e-12Construct a FiniteDifferences on a custom grid:julia> method, report = fdm([-2, 0, 5], 1, report=true)\n(FiniteDifferences.method, FiniteDifferencesReport:\n  order of method:       3\n  order of derivative:   1\n  grid:                  [-2, 0, 5]\n  coefficients:          [-0.357143, 0.3, 0.0571429]\n  roundoff error:        2.22e-16\n  bounds on derivatives: 1.00e+00\n  step size:             3.62e-06\n  accuracy:              6.57e-11\n)\n\njulia> method(sin, 1) - cos(1)\n-2.05648831297367e-11Compute a directional derivative:julia> f(x) = sum(x)\nf (generic function with 1 method)\n\njulia> central_fdm(5, 1)(ε -> f([1, 1, 1] + ε * [1, 2, 3]), 0) - 6\n-2.922107000813412e-13"
},

{
    "location": "pages/api.html#FiniteDifferences.assert_approx_equal-NTuple{5,Any}",
    "page": "API",
    "title": "FiniteDifferences.assert_approx_equal",
    "category": "method",
    "text": "assert_approx_equal(x, y, ε_abs, ε_rel[, desc])\n\nAssert that x is approximately equal to y.\n\nLet eps_z = eps_abs / eps_rel. Call x and y small if abs(x) < eps_z and abs(y) < eps_z, and call x and y large otherwise.  If this function returns True, then it is guaranteed that abs(x - y) < 2 eps_rel max(abs(x), abs(y)) if x and y are large, and abs(x - y) < 2 eps_abs if x and y are small.\n\nArguments\n\nx: First object to compare.\ny: Second object to compare.\nε_abs: Absolute tolerance.\nε_rel: Relative tolerance.\ndesc: Description of the comparison. Omit or set to false to have no description.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#FiniteDifferences.backward_fdm",
    "page": "API",
    "title": "FiniteDifferences.backward_fdm",
    "category": "function",
    "text": "FiniteDifferences.Backward(p, q; kwargs...)\nbackward_fdm(p, q; kwargs...)\n\nConstruct a backward finite difference method of order p to compute the qth derivative. See FiniteDifferenceMethod for more details.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#FiniteDifferences.central_fdm",
    "page": "API",
    "title": "FiniteDifferences.central_fdm",
    "category": "function",
    "text": "FiniteDifferences.Central(p, q; kwargs...)\ncentral_fdm(p, q; kwargs...)\n\nConstruct a central finite difference method of order p to compute the qth derivative. See FiniteDifferenceMethod for more details.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#FiniteDifferences.fdm-Union{Tuple{M}, Tuple{M,Any,Any,Val{true}}} where M<:FiniteDifferences.FiniteDifferenceMethod",
    "page": "API",
    "title": "FiniteDifferences.fdm",
    "category": "method",
    "text": "fdm(m::FiniteDifferenceMethod, f, x[, Val(false)]; kwargs...) -> Real\nfdm(m::FiniteDifferenceMethod, f, x, Val(true); kwargs...) -> Tuple{FiniteDifferenceMethod, Real}\n\nCompute the derivative of f at x using the finite differencing method m. The optional Val argument dictates whether the method should be returned alongside the derivative value, which can be useful for examining the step size used and other such parameters.\n\nThe recognized keywords are:\n\nadapt: The number of adaptive steps to use improve the estimate of bound.\nbound: Bound on the value of the function and its derivatives at x.\ncondition: The condition number. See DEFAULT_CONDITION.\neps: The assumed roundoff error. Defaults to eps() plus TINY.\n\nwarning: Warning\nBounds can\'t be adaptively computed over nonstandard grids; passing a value for adapt greater than 0 when m::Nonstandard results in an error.\n\nnote: Note\nCalling FiniteDifferenceMethod objects is equivalent to passing them to fdm.\n\nExamples\n\njulia> fdm(central_fdm(5, 1), sin, 1; adapt=2)\n0.5403023058681039\n\njulia> fdm(central_fdm(2, 1), exp, 0, Val(true))\n(FiniteDifferenceMethod:\n  order of method:       2\n  order of derivative:   1\n  grid:                  [-1, 1]\n  coefficients:          [-0.5, 0.5]\n  roundoff error:        1.42e-14\n  bounds on derivatives: 1.00e+02\n  step size:             1.69e-08\n  accuracy:              1.69e-06\n, 1.0000000031817473)\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#FiniteDifferences.forward_fdm",
    "page": "API",
    "title": "FiniteDifferences.forward_fdm",
    "category": "function",
    "text": "FiniteDifferences.Forward(p, q; kwargs...)\nforward_fdm(p, q; kwargs...)\n\nConstruct a forward finite difference method of order p to compute the qth derivative. See FiniteDifferenceMethod for more details.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": "Modules = [FiniteDifferences]\nPrivate = false"
},

]}
