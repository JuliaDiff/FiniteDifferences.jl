# FiniteDifferences.jl: Finite Difference Methods

[![CI](https://github.com/JuliaDiff/FiniteDifferences.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaDiff/FiniteDifferences.jl/actions?query=workflow%3ACI)
[![Build Status](https://travis-ci.org/JuliaDiff/FiniteDifferences.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/FiniteDifferences.jl)
[![codecov.io](https://codecov.io/github/JuliaDiff/FiniteDifferences.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaDiff/FiniteDifferences.jl?branch=master)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/F/FiniteDifferences.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)

[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliadiff.github.io/FiniteDifferences.jl/latest/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

FiniteDifferences.jl estimates derivatives with [finite differences](https://en.wikipedia.org/wiki/Finite_difference).

See also the Python package [FDM](https://github.com/wesselb/fdm).

#### FiniteDiff.jl vs FiniteDifferences.jl
[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) and [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
are similar libraries: both calculate approximate derivatives numerically.
You should definately use one or the other, rather than the legacy [Calculus.jl](https://github.com/JuliaMath/Calculus.jl) finite differencing, or reimplementing it yourself.
At some point in the future they might merge, or one might depend on the other.
Right now here are the differences:

 - FiniteDifferences.jl supports basically any type, where as FiniteDiff.jl supports only array-ish types
 - FiniteDifferences.jl supports higher order approximation
 - FiniteDiff.jl is carefully optimized to minimize allocations
 - FiniteDiff.jl supports coloring vectors for efficient calculation of sparse Jacobians


#### FDM.jl
This package was formerly called FDM.jl. We recommend users of FDM.jl [update to FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl/issues/37).


## Example: Scalar Derivatives

Compute the first derivative of `sin` with a 5th order central method:

```julia
julia> central_fdm(5, 1)(sin, 1) - cos(1)
-2.4313884239290928e-14
```

Compute the second derivative of `sin` with a 5th order central method:

```julia
julia> central_fdm(5, 2)(sin, 1) + sin(1)
-4.367495254342657e-11
```

The functions `forward_fdm` and `backward_fdm` can be used to construct
forward differences and backward differences respectively.

Alternatively, you can construct a finite difference method on a custom grid:

```julia
julia> method = FiniteDifferenceMethod([-2, 0, 5], 1)
FiniteDifferenceMethod:
  order of method:       3
  order of derivative:   1
  grid:                  [-2, 0, 5]
  coefficients:          [-0.35714285714285715, 0.3, 0.05714285714285714]

julia> method(sin, 1) - cos(1)
-8.423706177040913e-11
```

Compute a directional derivative:

```julia
julia> f(x) = sum(x)
f (generic function with 1 method)

julia> central_fdm(5, 1)(ε -> f([1, 1, 1] .+ ε .* [1, 2, 3]), 0) - 6
-6.217248937900877e-15
```

## Example: Multivariate Derivatives

Consider a quadratic function:

```julia
julia> a = randn(3, 3); a = a * a'
3×3 Array{Float64,2}:
  8.0663   -1.12965   1.68556
 -1.12965   3.55005  -3.10405
  1.68556  -3.10405   3.77251

julia> f(x) = 0.5 * x' * a * x
```

Compute the gradient:

```julia
julia> grad(central_fdm(5, 1), f, x)[1] - a * x
3-element Array{Float64,1}:
 -1.2612133559741778e-12
 -3.526068326209497e-13
 -2.3305801732931286e-12
```

Compute the Jacobian:

```julia
julia> jacobian(central_fdm(5, 1), f, x)[1] - (a * x)'
1×3 Array{Float64,2}:
 -1.26121e-12  -3.52607e-13  -2.33058e-12
```

The Jacobian can also be computed for non-scalar functions:

```julia
julia> a = randn(3, 3)
3×3 Array{Float64,2}:
 -0.343783   1.5708     0.723289
 -0.425706  -0.478881  -0.306881
  1.27326   -0.171606   2.23671

julia> f(x) = a * x

julia> jacobian(central_fdm(5, 1), f, x)[1] - a
3×3 Array{Float64,2}:
 -2.81331e-13   2.77556e-13  1.28342e-13
 -3.34732e-14  -6.05072e-15  6.05072e-15
 -2.24709e-13   1.88821e-13  1.06581e-14
```

To compute Jacobian--vector products, use `jvp` and `j′vp`:
 
```julia
julia> v = randn(3)
3-element Array{Float64,1}:
 -1.290782164377614
 -0.37701592844250903
 -1.4288108966380777

julia> jvp(central_fdm(5, 1), f, (x, v)) - a * v
3-element Array{Float64,1}:
 -1.3233858453531866e-13
  9.547918011776346e-15
  3.632649736573512e-13

julia> j′vp(central_fdm(5, 1), f, x, v)[1] - a'x
 3-element Array{Float64,1}:
  3.5704772471945034e-13
  4.04121180963557e-13
  1.2807532812075806e-12
```
