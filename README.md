# FDM.jl: Finite Difference Methods

[![Build Status](https://travis-ci.org/willtebbutt/FDM.jl.svg?branch=master)](https://travis-ci.org/willtebbutt/FDM.jl)
[![Coverage Status](https://coveralls.io/repos/willtebbutt/FDM.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/willtebbutt/FDM.jl?branch=master)
[![codecov.io](http://codecov.io/github/willtebbutt/FDM.jl/coverage.svg?branch=master)](http://codecov.io/github/willtebbutt/FDM.jl?branch=master)

FDM.jl approximates derivatives of functions using finite difference methods.

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

Construct a FDM on a custom grid:

```julia
julia> method, report = fdm([-2, 0, 5], 1, report=true)
(FDM.method, FDMReport:
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
