export FiniteDifferenceMethod, fdm, backward_fdm, forward_fdm, central_fdm, extrapolate_fdm

"""
    add_tiny(x::Real)

Add a tiny number, 10^{-40}, to `x`, preserving the type. If `x` is an `Integer`, it is
promoted to a suitable floating point type.
"""
add_tiny(x::T) where T<:Real = x + convert(T, 1e-40)
add_tiny(x::Integer) = add_tiny(float(x))

"""
    FiniteDifferences.DEFAULT_CONDITION

The default [condition number](https://en.wikipedia.org/wiki/Condition_number) used when
computing bounds. It provides amplification of the ∞-norm when passed to the function's
derivatives.
"""
const DEFAULT_CONDITION = 100

"""
    FiniteDifferenceMethod{G<:AbstractVector, C<:AbstractVector}

A finite difference method.

# Fields
- `grid::G`: Multiples of the step size that the function will be evaluated at.
- `q::Int`: Order of derivative to estimate.
- `coefs::C`: Coefficients corresponding to the grid functions that the function evaluations
    will be weighted by.
- `bound_estimator::Function`: A function that takes in the function and the evaluation
    point and returns a bound on the magnitude of the `length(grid)`th derivative.
"""
struct FiniteDifferenceMethod{G<:AbstractVector, C<:AbstractVector, E<:Function}
    grid::G
    q::Int
    coefs::C
    bound_estimator::E
end

"""
    FiniteDifferenceMethod(
        grid::AbstractVector,
        q::Int;
        condition::Int=DEFAULT_CONDITION
    )

Construct a finite difference method.

# Arguments
- `grid::Abstract`: The grid. See [`FiniteDifferenceMethod`](@ref).
- `q::Int`: Order of the derivative to estimate.
- `condition::Int`: Condition number. See [`DEFAULT_CONDITION`](@ref).

# Returns
- `FiniteDifferenceMethod`: Specified finite difference method.
"""
function FiniteDifferenceMethod(
    grid::AbstractVector,
    q::Int;
    condition::Int=DEFAULT_CONDITION
)
    p = length(grid)
    _check_p_q(p, q)
    return FiniteDifferenceMethod(
        grid,
        q,
        _coefs(grid, q),
        _make_default_bound_estimator(condition=condition)
    )
end

"""
    (m::FiniteDifferenceMethod)(
        f::Function,
        x::T;
        factor::Int=1,
        max_step=convert(T, 0.1)
    ) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and an
automatically determined step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.

# Keywords
- `factor::Int=1`: Factor to amplify the estimated round-off error by. This can be used
    to force a more conservative step size.
- `max_step=convert(T, 0.1)`: Maximum step size.

# Returns
- Estimate of the derivative.

# Examples

```julia-repl
julia> fdm = central_fdm(5, 1)
FiniteDifferenceMethod:
  order of method:       5
  order of derivative:   1
  grid:                  [-2, -1, 0, 1, 2]
  coefficients:          [0.08333333333333333, -0.6666666666666666, 0.0, 0.6666666666666666, -0.08333333333333333]

julia> fdm(sin, 1)
0.5403023058681155

julia> fdm(sin, 1) - cos(1)  # Check the error.
-2.4313884239290928e-14

julia> FiniteDifferences.estimate_step(fdm, sin, 1.0)  # Computes step size and estimates the error.
(0.0010632902144695163, 1.9577610541734626e-13)
```
"""
function (m::FiniteDifferenceMethod)(
    f::Function,
    x::T;
    factor::Int=1,
    max_step=convert(T, 0.1)
) where T<:AbstractFloat
    # The automatic step size calculation fails if `m.q == 0`, so handle that edge case.
    iszero(m.q) && return f(x)
    h, _ = estimate_step(m, f, x, factor=factor, max_step=max_step)
    return m(f, x, h)
end
# Handle arguments that are not floats. Assume that converting to float is desired.
function (m::FiniteDifferenceMethod)(f::Function, x::T; kw_args...) where T<:Real
    return m(f, float(x); kw_args...)
end

"""
    (m::FiniteDifferenceMethod)(
        f::Function,
        x::T,
        h
    ) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and a given
step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.
- `h`: Step size.

# Returns
- Estimate of the derivative.

# Examples

```julia-repl
julia> fdm = central_fdm(5, 1)
FiniteDifferenceMethod:
  order of method:       5
  order of derivative:   1
  grid:                  [-2, -1, 0, 1, 2]
  coefficients:          [0.08333333333333333, -0.6666666666666666, 0.0, 0.6666666666666666, -0.08333333333333333]

julia> fdm(sin, 1, 1e-3)
0.5403023058679624

julia> fdm(sin, 1, 1e-3) - cos(1)  # Check the error.
-1.7741363933510002e-13
```
"""
function (m::FiniteDifferenceMethod)(
    f::Function,
    x::T,
    h
) where T<:AbstractFloat
    return sum(
        i -> convert(T, m.coefs[i]) * f(T(x + h * m.grid[i])),
        eachindex(m.grid)
    ) / h^m.q
end
# Handle arguments that are not floats. Assume that converting to float is desired.
(m::FiniteDifferenceMethod)(f::Function, x::T, h) where T<:Real = m(f, float(x), h)

# Check the method and derivative orders for consistency.
function _check_p_q(p::Integer, q::Integer)
    q >= 0 || throw(ArgumentError("order of derivative must be non-negative"))
    q < p || throw(ArgumentError("order of the method must be strictly greater than that " *
                                 "of the derivative"))
    # Check whether the method can be computed. We require the factorial of the method order
    # to be computable with regular `Int`s, but `factorial` will after 20, so 20 is the
    # largest we can allow.
    p > 20 && throw(ArgumentError("order of the method is too large to be computed"))
    return
end

const _COEFFS_CACHE = Dict{Tuple{AbstractVector{<:Real}, Integer}, Vector{Float64}}()

# Compute coefficients for the method and cache the result.
function _coefs(grid::AbstractVector{<:Real}, q::Integer)
    return get!(_COEFFS_CACHE, (grid, q)) do
        p = length(grid)
        # For high precision on the `\`, we use `Rational`, and to prevent overflows we use
        # `Int128`. At the end we go to `Float64` for fast floating point math, rather than
        # rational math.
        C = [Rational{Int128}(g^i) for i in 0:(p - 1), g in grid]
        x = zeros(Rational{Int128}, p)
        x[q + 1] = factorial(q)
        return Float64.(C \ x)
    end
end

# Estimate the bound on the derivative by amplifying the ∞-norm.
function _make_default_bound_estimator(; condition::Int=DEFAULT_CONDITION)
    return (f, x) -> condition * maximum(abs.(f(x)))
end

function Base.show(io::IO, x::FiniteDifferenceMethod)
    @printf io "FiniteDifferenceMethod:\n"
    @printf io "  order of method:       %d\n" length(x.grid)
    @printf io "  order of derivative:   %d\n" x.q
    @printf io "  grid:                  %s\n" x.grid
    @printf io "  coefficients:          %s\n" x.coefs
end

"""
    function estimate_step(
        m::FiniteDifferenceMethod,
        f::Function,
        x::T;
        factor::Int=1,
        max_step=convert(T, 0.1)
    ) where T<:AbstractFloat

Estimate the step size for a finite difference method `m`. Also estimates the error of the
estimate of the derivative.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.

# Keywords
- `factor::Int=1`. Factor to amplify the estimated round-off error by. This can be used
    to force a more conservative step size.
- `max_step=convert(T, 0.1)`: Maximum step size.

# Returns
- `Tuple{T, T}`: Estimated step size and an estimate of the error of the finite difference
    estimate.
"""
function estimate_step(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    factor::Int=1,
    max_step=convert(T, 0.1)
) where T<:AbstractFloat
    p = length(m.coefs)
    q = m.q
    f_x = float(f(x))

    # Estimate the bound and round-off error.
    ε = add_tiny(maximum(eps.(f_x))) * factor
    M = add_tiny(m.bound_estimator(f, x))

    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = ε * sum(abs, m.coefs)
    C₂ = M * sum(n -> abs(m.coefs[n] * m.grid[n]^p), eachindex(m.coefs)) / factorial(p)
    h = convert(T, min((q / (p - q) * C₁ / C₂)^(1 / p), max_step))

    # Estimate the accuracy of the method.
    accuracy = h^(-q) * C₁ + h^(p - q) * C₂

    return h, accuracy
end

for direction in [:forward, :central, :backward]
    fdm_fun = Symbol(direction, "_fdm")
    grid_fun = Symbol("_", direction, "_grid")
    @eval begin function $fdm_fun(
            p::Int,
            q::Int;
            adapt::Int=1,
            condition::Int=DEFAULT_CONDITION,
            geom::Bool=false
        )
            _check_p_q(p, q)
            grid = collect($grid_fun(p))
            geom && (grid = _exponentiate_grid(grid))
            coefs = _coefs(grid, q)
            return FiniteDifferenceMethod(
                grid,
                q,
                coefs,
                _make_adaptive_bound_estimator(
                    $fdm_fun,
                    p,
                    adapt,
                    condition,
                    geom=geom
                )
            )
        end

        @doc """
    $($(Meta.quot(fdm_fun)))(
        p::Int,
        q::Int;
        adapt::Int=1,
        condition::Int=DEFAULT_CONDITION,
        geom::Bool=false
    )

Contruct a finite difference method at a $($(Meta.quot(direction))) grid of `p` linearly
spaced points.

# Arguments
- `p::Int`: Number of grid points.
- `q::Int`: Order of the derivative to estimate.

# Keywords
- `adapt::Int=1`: Use another finite difference method to estimate the magnitude of the
    `p`th order derivative, which is important for the step size computation. Recurse
    this procedure `adapt` times.
- `condition::Int`: Condition number. See [`DEFAULT_CONDITION`](@ref).
- `geom::Bool`: Use geometrically spaced points instead of linearly spaced points.

# Returns
- `FiniteDifferenceMethod`: The specified finite difference method.
        """ $fdm_fun
    end
end

function _make_adaptive_bound_estimator(
    constructor::Function,
    q::Int,
    adapt::Int,
    condition::Int;
    kw_args...
)
    if adapt >= 1
        estimate_derivative =
            constructor(q + 1, q, adapt=adapt - 1, condition=condition; kw_args...)
        return (f, x) -> maximum(abs.(estimate_derivative(f, x)))
    else
        return _make_default_bound_estimator(condition=condition)
    end
end

_forward_grid(p::Int) = 0:(p - 1)

_backward_grid(p::Int) = (1 - p):0

function _central_grid(p::Int)
    if isodd(p)
        return div(1 - p, 2):div(p - 1, 2)
    else
        return vcat(div(-p, 2):-1, 1:div(p, 2))
    end
end

_exponentiate_grid(grid::Vector, base::Int=3) = sign.(grid) .* base .^ abs.(grid) ./ base

function _is_symmetric(vec::Vector; centre_zero=false, negate_half=false)
    n = div(length(vec), 2)
    half_sign = negate_half ? -1 : 1
    if isodd(length(vec))
        centre_zero && vec[n + 1] != 0 && return false
        return vec[1:n] == half_sign .* reverse(vec[n + 2:end])
    else
        return vec[1:n] == half_sign .* reverse(vec[n + 1:end])
    end
end

function _is_symmetric(m::FiniteDifferenceMethod)
    grid_symmetric = _is_symmetric(m.grid, centre_zero=true, negate_half=true)
    coefs_symmetric =_is_symmetric(m.coefs, negate_half=true)
    return grid_symmetric && coefs_symmetric
end

"""
    extrapolate_fdm(
        m::FiniteDifferenceMethod,
        f::Function,
        x::T;
        factor::Int=1000,
        kw_args...
    ) where T<:AbstractFloat

Use Richardson extrapolation to refine a finite difference method. This method uses
[`estimate_step`](@ref) to determine an appropriate initial step size for
`Richardson.extrapolate`.

Takes further in keyword arguments for `Richardson.extrapolate`. This method
automatically sets `power = 2` if `m` is symmetric.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.

# Factor
- `factor::Int=1000`: Factor to amplify the estimated step size by.

# Returns
- `Tuple{Real, Real}`: Estimate of the derivative and error.
"""
function extrapolate_fdm(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    factor::Int=1000,
    kw_args...
) where T<:AbstractFloat
    h_conservative = estimate_step(m, f, x)[1] * factor
    return extrapolate_fdm(m, f, x, h_conservative; kw_args...)
end
# Handle arguments that are not floats. Assume that converting to float is desired.
function extrapolate_fdm(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    kw_args...
) where T<:Real
    return extrapolate_fdm(m, f, float(x); kw_args...)
end

"""
    extrapolate_fdm(
        m::FiniteDifferenceMethod,
        f::Function,
        x::T,
        h;
        kw_args...
    ) where T<:AbstractFloat

Use Richardson extrapolation to refine a finite difference method. This method requires
a given initial step size for `Richardson.extrapolate`.

Takes further in keyword arguments for `Richardson.extrapolate`. This method
automatically sets `power = 2` if `m` is symmetric.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.
- `h`: Initial step size.

# Returns
- `Tuple{Real, Real}`: Estimate of the derivative and error.
"""
function extrapolate_fdm(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T,
    h;
    kw_args...
) where T<:AbstractFloat
    if _is_symmetric(m)
        power = 2
    else
        power = 1
    end
    return extrapolate(h -> m(f, x, h), h; power=power, kw_args...)
end
# Handle arguments that are not floats. Assume that converting to float is desired.
function extrapolate_fdm(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T,
    h;
    kw_args...
) where T<:Real
    return extrapolate_fdm(m, f, float(x), h; kw_args...)
end
