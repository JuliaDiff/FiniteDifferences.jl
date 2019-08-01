export fdm, backward_fdm, forward_fdm, central_fdm

"""
    FiniteDifferences.DEFAULT_CONDITION

The default [condition number](https://en.wikipedia.org/wiki/Condition_number) used when
computing bounds. It provides amplification of the ∞-norm when passed to the function's
derivatives.
"""
const DEFAULT_CONDITION = 100

"""
    FiniteDifferences.TINY

A tiny number added to some quantities to ensure that division by 0 does not occur.
"""
const TINY = 1e-20

forward_grid(p::Int) = 0:(p - 1)
backward_grid(p::Int) = (1 - p):0
function central_grid(p::Int)
    if isodd(p)
        return div(1 - p, 2):div(p - 1, 2)
    else
        return vcat(div(-p, 2):-1, 1:div(p, 2))
    end
end

"""
    History

A mutable type that tracks several values during adaptive bound computation.
"""
mutable struct History
    adapt::Int
    eps::Real
    bound::Real
    step::Real
    accuracy::Real

    function History(; kwargs...)
        h = new()
        for (k, v) in kwargs
            setfield!(h, k, v)
        end
        return h
    end
end

"""
    FiniteDifferenceMethod

Abstract type for all finite differencing method types.
Subtypes of `FiniteDifferenceMethod` are callable with the signature

```
method(f, x; kwargs...)
```

where the keyword arguments can be any of

* `adapt`: The number of adaptive steps to use improve the estimate of `bound`.
* `bound`: Bound on the value of the function and its derivatives at `x`.
* `condition`: The condition number. See [`DEFAULT_CONDITION`](@ref).
* `eps`: The assumed roundoff error. Defaults to `eps()` plus [`TINY`](@ref).
"""
abstract type FiniteDifferenceMethod end

function Base.show(io::IO, x::FiniteDifferenceMethod)
    @printf io "FiniteDifferenceMethod:\n"
    @printf io "  order of method:       %d\n" x.p
    @printf io "  order of derivative:   %d\n" x.q
    @printf io "  grid:                  %s\n" x.grid
    @printf io "  coefficients:          %s\n" x.coefs
    h = x.history
    if all(p->isdefined(h, p), propertynames(h))
        @printf io "  roundoff error:        %.2e\n" h.eps
        @printf io "  bounds on derivatives: %.2e\n" h.bound
        @printf io "  step size:             %.2e\n" h.step
        @printf io "  accuracy:              %.2e\n" h.accuracy
    end
end

for D in (:Forward, :Backward, :Central, :Nonstandard)
    @eval begin
        struct $D{G<:AbstractVector, C<:AbstractVector} <: FiniteDifferenceMethod
            p::Int
            q::Int
            grid::G
            coefs::C
            history::History
        end
        (d::$D)(f, x=0.0; kwargs...) = fdm(d, f, x; kwargs...)
    end
end

# The below does not apply to Nonstandard, as it has its own constructor
for D in (:Forward, :Backward, :Central)
    lcname = lowercase(String(D))
    gridf = Symbol(lcname, "_grid")
    fdmf = Symbol(lcname, "_fdm")

    @eval begin
        # Compatibility layer over the "old" API
        function $fdmf(p::Integer, q::Integer; adapt=1, kwargs...)
            _dep_kwarg(kwargs)
            return $D(p, q; adapt=adapt, kwargs...)
        end

        function $D(p::Integer, q::Integer; adapt=1, kwargs...)
            _check_p_q(p, q)
            grid = $gridf(p)
            coefs = _coefs(grid, p, q)
            hist = History(; adapt=adapt, kwargs...)
            return $D{typeof(grid), typeof(coefs)}(Int(p), Int(q), grid, coefs, hist)
        end

        @doc """
            FiniteDifferences.$($(Meta.quot(D)))(p, q; kwargs...)
            $($(Meta.quot(fdmf)))(p, q; kwargs...)

        Construct a $($lcname) finite difference method of order `p` to compute the `q`th
        derivative.
        See [`FiniteDifferenceMethod`](@ref) for more details.
        """
        ($D, $fdmf)
    end
end

"""
    FiniteDifferences.Nonstandard(grid, q; kwargs...)

An finite differencing method which is constructed based on a user-defined grid. It is
nonstandard in the sense that it represents neither forward, backward, nor central
differencing.
See [`FiniteDifferenceMethod`](@ref) for further details.
"""
function Nonstandard(grid::AbstractVector{<:Real}, q::Integer; adapt=0, kwargs...)
    p = length(grid)
    _check_p_q(p, q)
    coefs = _coefs(grid, p, q)
    hist = History(; adapt=adapt, kwargs...)
    return Nonstandard{typeof(grid), typeof(coefs)}(Int(p), Int(q), grid, coefs, hist)
end

# Check the method and derivative orders for consistency
function _check_p_q(p::Integer, q::Integer)
    q < p || throw(ArgumentError("order of the method must be strictly greater than that " *
                                 "of the derivative"))
    # Check whether the method can be computed. We require the factorial of the
    # method order to be computable with regular `Int`s, but `factorial` will overflow
    # after 20, so 20 is the largest we can allow.
    p > 20 && throw(ArgumentError("order of the method is too large to be computed"))
    return
end

# Compute coefficients for the method
function _coefs(grid::AbstractVector{<:Real}, p::Integer, q::Integer)
    C = [g^i for i in 0:(p - 1), g in grid]
    x = zeros(Int, p)
    x[q + 1] = factorial(q)
    return C \ x
end

# Estimate the bound on the function value and its derivatives at a point
_estimate_bound(x, cond) = cond * maximum(abs, x) + TINY

"""
    fdm(m::FiniteDifferenceMethod, f, x[, Val(false)]; kwargs...) -> Real
    fdm(m::FiniteDifferenceMethod, f, x, Val(true); kwargs...) -> Tuple{FiniteDifferenceMethod, Real}

Compute the derivative of `f` at `x` using the finite differencing method `m`.
The optional `Val` argument dictates whether the method should be returned alongside the
derivative value, which can be useful for examining the step size used and other such
parameters.

The recognized keywords are:

* `adapt`: The number of adaptive steps to use improve the estimate of `bound`.
* `bound`: Bound on the value of the function and its derivatives at `x`.
* `condition`: The condition number. See [`DEFAULT_CONDITION`](@ref).
* `eps`: The assumed roundoff error. Defaults to `eps()` plus [`TINY`](@ref).

!!! warning
    Bounds can't be adaptively computed over nonstandard grids; passing a value for
    `adapt` greater than 0 when `m::Nonstandard` results in an error.

!!! note
    Calling [`FiniteDifferenceMethod`](@ref) objects is equivalent to passing them to `fdm`.

# Examples

```julia-repl
julia> fdm(central_fdm(5, 1), sin, 1; adapt=2)
0.5403023058681039

julia> fdm(central_fdm(2, 1), exp, 0, Val(true))
(FiniteDifferenceMethod:
  order of method:       2
  order of derivative:   1
  grid:                  [-1, 1]
  coefficients:          [-0.5, 0.5]
  roundoff error:        1.42e-14
  bounds on derivatives: 1.00e+02
  step size:             1.69e-08
  accuracy:              1.69e-06
, 1.0000000031817473)
```
"""
function fdm(
    m::M,
    f,
    x,
    ::Val{true};
    condition=DEFAULT_CONDITION,
    bound=_estimate_bound(f(x), condition),
    eps=(Base.eps(float(bound)) + TINY),
    adapt=m.history.adapt,
    max_step=0.1,
) where M<:FiniteDifferenceMethod
    if M <: Nonstandard && adapt > 0
        throw(ArgumentError("can't adaptively compute bounds over Nonstandard grids"))
    end
    eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
    bound > 0 || throw(ArgumentError("bound must be positive, got $bound"))
    0 <= adapt < 20 - m.p || throw(ArgumentError("can't perform $adapt adaptation steps"))

    p = m.p
    q = m.q
    grid = m.grid
    coefs = m.coefs

    # Adaptively compute the bound on the function and derivative values, if applicable.
    if adapt > 0
        newm = (M.name.wrapper)(p + 1, p)
        dfdx = fdm(
            newm,
            f,
            x;
            condition=condition,
            eps=eps,
            bound=bound,
            max_step=max_step,
            adapt=(adapt - 1),
        )
        bound = _estimate_bound(dfdx, condition)
    end

    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = eps * sum(abs, coefs)
    C₂ = bound * sum(n->abs(coefs[n] * grid[n]^p), eachindex(coefs)) / factorial(p)
    ĥ = min((q / (p - q) * C₁ / C₂)^(1 / p), max_step)

    # Estimate the accuracy of the method.
    accuracy = ĥ^(-q) * C₁ + ĥ^(p - q) * C₂

    # Estimate the value of the derivative.
    dfdx = sum(i->coefs[i] * f(x + ĥ * grid[i]), eachindex(grid)) / ĥ^q

    m.history.eps = eps
    m.history.bound = bound
    m.history.step = ĥ
    m.history.accuracy = accuracy

    return m, dfdx
end

function fdm(m::FiniteDifferenceMethod, f, x, ::Val{false}=Val(false); kwargs...)
    _, dfdx = fdm(m, f, x, Val(true); kwargs...)
    return dfdx
end


## Deprecations

# Used for keyword argument name deprecations
function _dep_kwarg(kwargs)
    for (old, new) in [(:ε, :eps), (:M, :bound)]
        haskey(kwargs, old) || continue
        val = kwargs[old]
        error(
            "keyword argument `", old, "` should now be passed as `", new, "` upon ",
            "application of the method. For example:\n    ",
            "central_fdm(5, 1)(f, x; $new=$val)\n",
            "not\n    ",
            "central_fdm(5, 1; $old=$val)(f, x)"
        )
    end
end

function fdm(
    grid::AbstractVector{<:Real},
    q::Int,
    ::Union{Val{true}, Val{false}}=Val(false);
    kwargs...,
)
    error("to use a custom grid, use `Nonstandard(grid, q)` and pass the result to `fdm`")
end

for fdmf in (:central_fdm, :backward_fdm, :forward_fdm)
    @eval function $fdmf(p::Int, q::Int, ::Union{Val{true}, Val{false}}; kwargs...)
        error(
            "the `Val` argument should now be passed directly to `fdm` after ",
            "constructing the method, not to the method constructor itself"
        )
    end
end
