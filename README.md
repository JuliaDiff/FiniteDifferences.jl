# FiniteDifferences.jl: Finite Difference Methods

[![Build Status](https://travis-ci.org/JuliaDiff/FiniteDifferences.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/FiniteDifferences.jl)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/g0gun5dxbkt631am/branch/master?svg=true)](https://ci.appveyor.com/project/JuliaDiff/FiniteDifferences-jl/branch/master)
[![codecov.io](http://codecov.io/github/JuliaDiff/FiniteDifferences.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiff/FiniteDifferences.jl?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliadiff.github.io/FiniteDifferences.jl/latest/)

FiniteDifferences.jl estimates derivatives with [finite differences](https://en.wikipedia.org/wiki/Finite_difference).

See also the Python package [FDM](https://github.com/wesselb/fdm).

## Examples

Compute the first derivative of `sin` with a 5th order central method:

```julia
julia> central_fdm(5, 1)(sin, 1) - cos(1)
-1.247890679678676e-13
```
Compute the second derivative of `sin` with a 5th order central method:

```julia
julia> central_fdm(5, 2)(sin, 1) + sin(1)
9.747314066999024e-12
```

Construct a FiniteDifferences on a custom grid:

```julia
julia> method, report = fdm([-2, 0, 5], 1, report=true)
(FiniteDifferences.method, FiniteDifferencesReport:
  order of method:       3
  order of derivative:   1
  grid:                  [-2, 0, 5]
  coefficients:          [-0.357143, 0.3, 0.0571429]
  roundoff error:        2.22e-16
  bounds on derivatives: 1.00e+00
  step size:             3.62e-06
  accuracy:              6.57e-11
)

julia> method(sin, 1) - cos(1)
-2.05648831297367e-11
```

Compute a directional derivative:

```julia
julia> f(x) = sum(x)
f (generic function with 1 method)

julia> central_fdm(5, 1)(ε -> f([1, 1, 1] + ε * [1, 2, 3]), 0) - 6
-2.922107000813412e-13
```

## FDM.jl

This package was formerly called FDM.jl. We recommend users of FDM.jl [update to FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl/issues/37).
