# FiniteDifferences.jl: Finite Difference Methods

[![CI](https://github.com/JuliaDiff/FiniteDifferences.jl/workflows/CI/badge.svg?branch=master)](https://github.com/JuliaDiff/FiniteDifferences.jl/actions?query=workflow%3ACI)
[![Build Status](https://travis-ci.org/JuliaDiff/FiniteDifferences.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/FiniteDifferences.jl)
[![codecov.io](https://codecov.io/github/JuliaDiff/FiniteDifferences.jl/coverage.svg?branch=master)](https://codecov.io/github/JuliaDiff/FiniteDifferences.jl?branch=master)
[![PkgEval](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/F/FiniteDifferences.svg)](https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html)

[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliadiff.org/FiniteDifferences.jl/latest/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/117872213.svg)](https://zenodo.org/badge/latestdoi/117872213)

FiniteDifferences.jl estimates derivatives with [finite differences](https://en.wikipedia.org/wiki/Finite_difference).

See also the Python package [FDM](https://github.com/wesselb/fdm).

#### FiniteDiff.jl vs FiniteDifferences.jl
[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) and [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
are similar libraries: both calculate approximate derivatives numerically.
You should definately use one or the other, rather than the legacy [Calculus.jl](https://github.com/JuliaMath/Calculus.jl) finite differencing, or reimplementing it yourself.
At some point in the future they might merge, or one might depend on the other.
Right now here are the differences:

 - FiniteDifferences.jl supports basically any type, where as FiniteDiff.jl supports only array-ish types
 - FiniteDifferences.jl supports higher order approximation and step size adaptation
 - FiniteDiff.jl supports caching and in-place computation
 - FiniteDiff.jl supports coloring vectors for efficient calculation of sparse Jacobians

#### FDM.jl
This package was formerly called FDM.jl. We recommend users of FDM.jl [update to FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl/issues/37).

#### Contents

* [Scalar Derivatives](#scalar-derivatives)
    - [Dealing with Singularities](#dealing-with-singularities)
    - [Richardson Extrapolation](#richardson-extrapolation)
    - [A Finite Difference Method on a Custom Grid](#a-finite-difference-method-on-a-custom-grid)
* [Multivariate Derivatives](#multivariate-derivatives)

## Scalar Derivatives

Compute the first derivative of `sin` with a 5th order central method:

```julia
julia> central_fdm(5, 1)(sin, 1) - cos(1)
-2.4313884239290928e-14
```

Finite difference methods are optimised to minimise allocations:

```julia
julia> m = central_fdm(5, 1);

julia> @benchmark $m(sin, 1)
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     486.621 ns (0.00% GC)
  median time:      507.677 ns (0.00% GC)
  mean time:        539.806 ns (0.00% GC)
  maximum time:     1.352 μs (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     195
```

Compute the second derivative of `sin` with a 5th order central method:

```julia
julia> central_fdm(5, 2)(sin, 1) - (-sin(1))
-8.767431225464861e-11
```

To obtain better accuracy, you can increase the order of the method:

```julia
julia> central_fdm(12, 2)(sin, 1) - (-sin(1))
5.240252676230739e-14
```

The functions `forward_fdm` and `backward_fdm` can be used to construct
forward differences and backward differences respectively.

### Dealing with Singularities

The function `log(x)` is only defined for `x > 0`.
If we try to use `central_fdm` to estimate the derivative of `log` near `x = 0`,
then we run into `DomainError`s, because `central_fdm` happens to evaluate `log`
at some `x < 0`.

```julia
julia> central_fdm(5, 1)(log, 1e-3)
ERROR: DomainError with -0.02069596546590111
```

To deal with this situation, you have two options.

The first option is to use `forward_fdm`, which only evaluates `log` at inputs
greater or equal to `x`:

```julia
julia> forward_fdm(5, 1)(log, 1e-3) - 1000
-3.741856744454708e-7
```

This works fine, but the downside is that you're restricted to one-sided
methods (`forward_fdm`), which tend to perform worse than central methods
(`central_fdm`).

The second option is to limit the distance that the finite difference method is
allowed to evaluate `log` away from `x`. Since `x = 1e-3`, a reasonable value
for this limit is `9e-4`:

```julia
julia> central_fdm(5, 1, max_range=9e-4)(log, 1e-3) - 1000
-4.027924660476856e-10
```

Another commonly encountered example is `logdet`, which is only defined
for positive-definite matrices.
Here you can use a forward method in combination with a positive-definite
deviation from `x`:

```julia
julia> x = diagm([1.0, 2.0, 3.0]); v = Matrix(1.0I, 3, 3);

julia> forward_fdm(5, 1)(ε -> logdet(x .+ ε .* v), 0) - sum(1 ./ diag(x))
-4.222400207254395e-12
```

A range-limited central method is also possible:

```julia
julia> central_fdm(5, 1, max_range=9e-1)(ε -> logdet(x .+ ε .* v), 0) - sum(1 ./ diag(x))
-1.283417816466681e-13
```

### Richardson Extrapolation

The finite difference methods defined in this package can be extrapolated using
[Richardson extrapolation](https://github.com/JuliaMath/Richardson.jl).
This can offer superior numerical accuracy:
Richardson extrapolation attempts polynomial extrapolation of the finite
difference estimate as a function of the step size until a convergence criterion
is reached.

```julia
julia> extrapolate_fdm(central_fdm(2, 1), sin, 1)[1] - cos(1)
1.6653345369377348e-14
```

Similarly, you can estimate higher order derivatives:

```julia
julia> extrapolate_fdm(central_fdm(5, 4), sin, 1)[1] - sin(1)
-1.626274487942503e-5
```

In this case, the accuracy can be improved by making the
[contraction rate](https://github.com/JuliaMath/Richardson.jl#usage) closer to
`1`:

```julia
julia> extrapolate_fdm(central_fdm(5, 4), sin, 1, contract=0.8)[1] - sin(1)
2.0725743343774639e-10
```

This performs similarly to a `10`th order central method:

```julia
julia> central_fdm(10, 4)(sin, 1) - sin(1)
-1.0301381969668455e-10
```

If you really want, you can even extrapolate the `10`th order central method,
but that provides no further gains:

```julia
julia> extrapolate_fdm(central_fdm(10, 4), sin, 1, contract=0.8)[1] - sin(1)
5.673617131662922e-10
```

In this case, the central method can be pushed to a high order to obtain
improved accuracy:

```julia
julia> central_fdm(20, 4)(sin, 1) - sin(1)
-5.2513549064769904e-14
```

### A Finite Difference Method on a Custom Grid

```julia
julia> method = FiniteDifferenceMethod([-2, 0, 5], 1)
FiniteDifferenceMethod:
  order of method:       3
  order of derivative:   1
  grid:                  [-2, 0, 5]
  coefficients:          [-0.35714285714285715, 0.3, 0.05714285714285714]

julia> method(sin, 1) - cos(1)
-3.701483564100272e-13
```

## Multivariate Derivatives

Consider a quadratic function:

```julia
julia> a = randn(3, 3); a = a * a'
3×3 Matrix{Float64}:
  4.31995    -2.80614   0.0829128
 -2.80614     3.91982   0.764388
  0.0829128   0.764388  1.18055

julia> f(x) = 0.5 * x' * a * x
```

Compute the gradient:

```julia
julia> x = randn(3)
3-element Vector{Float64}:
 -0.18563161988700727
 -0.4659836395595666
  2.304584409826511

julia> grad(central_fdm(5, 1), f, x)[1] - a * x
3-element Vector{Float64}:
  4.1744385725905886e-14
 -6.611378111642807e-14
 -8.615330671091215e-14
```

Compute the Jacobian:

```julia
julia> jacobian(central_fdm(5, 1), f, x)[1] - (a * x)'
1×3 Matrix{Float64}:
 4.17444e-14  -6.61138e-14  -8.61533e-14
```

The Jacobian can also be computed for non-scalar functions:

```julia
julia> a = randn(3, 3)
3×3 Matrix{Float64}:
  0.844846   1.04772    1.0173
 -0.867721   0.154146  -0.938077
  1.34078   -0.630105  -1.13287

julia> f(x) = a * x

julia> jacobian(central_fdm(5, 1), f, x)[1] - a
3×3 Matrix{Float64}:
  2.91989e-14   1.77636e-15   4.996e-14
 -5.55112e-15  -7.63278e-15   2.4758e-14
  4.66294e-15  -2.05391e-14  -1.04361e-14
```

To compute Jacobian--vector products, use `jvp` and `j′vp`:

```julia
julia> v = randn(3)
3-element Array{Float64,1}:
 -1.290782164377614
 -0.37701592844250903
 -1.4288108966380777

julia> jvp(central_fdm(5, 1), f, (x, v)) - a * v
3-element Vector{Float64}:
 -7.993605777301127e-15
 -8.881784197001252e-16
 -3.22519788653608e-14

julia> j′vp(central_fdm(5, 1), f, x, v)[1] - a'x
3-element Vector{Float64}:
 -2.1316282072803006e-14
  2.4646951146678475e-14
  6.661338147750939e-15
```
