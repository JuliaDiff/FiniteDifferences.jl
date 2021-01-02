export FiniteDifferenceMethod, fdm, backward_fdm, forward_fdm, central_fdm, extrapolate_fdm

"""
    FiniteDifferences.DEFAULT_CONDITION

The default value for the [condition number](https://en.wikipedia.org/wiki/Condition_number)
of finite difference method. The condition number specifies the amplification of the ∞-norm
when passed to the function's derivative.
"""
const DEFAULT_CONDITION = 10

"""
    FiniteDifferences.DEFAULT_FACTOR

The default factor number. The factor number specifies the multiple to amplify the estimated
round-off errors by.
"""
const DEFAULT_FACTOR = 1

abstract type FiniteDifferenceMethod end

"""
    UnadaptedFiniteDifferenceMethod{P,Q} <: FiniteDifferenceMethod

A finite difference method that estimates a `Q`th order derivative from `P` function
evaluations. This method does not dynamically adapt its step size.

# Fields
- `grid::NTuple{P,Float64}`: Multiples of the step size that the function will be evaluated
    at.
- `coefs::NTuple{P,Float64}`: Coefficients corresponding to the grid functions that the
    function evaluations will be weighted by.
- `bound_estimator::FiniteDifferenceMethod`: A finite difference method that is tuned to
    perform adaptation for this finite difference method.
- `condition::Float64`: Condition number. See See [`DEFAULT_CONDITION`](@ref).
- `factor::Float64`: Factor number. See See [`DEFAULT_FACTOR`](@ref).
"""
struct UnadaptedFiniteDifferenceMethod{P,Q} <: FiniteDifferenceMethod
    grid::NTuple{P,Float64}
    coefs::NTuple{P,Float64}
    condition::Float64
    factor::Float64
end

"""
    AdaptedFiniteDifferenceMethod{
        P,
        Q,
        E<:FiniteDifferenceMethod
    } <: FiniteDifferenceMethod

A finite difference method that estimates a `Q`th order derivative from `P` function
evaluations. This method does dynamically adapt its step size.

# Fields
- `grid::NTuple{P,Float64}`: Multiples of the step size that the function will be evaluated
    at.
- `coefs::NTuple{P,Float64}`: Coefficients corresponding to the grid functions that the
    function evaluations will be weighted by.
- `condition::Float64`: Condition number. See See [`DEFAULT_CONDITION`](@ref).
- `factor::Float64`: Factor number. See See [`DEFAULT_FACTOR`](@ref).
- `bound_estimator::E`: A finite difference method that is tuned to perform adaptation for
    this finite difference method.
"""
struct AdaptedFiniteDifferenceMethod{
    P,
    Q,
    E<:FiniteDifferenceMethod
} <: FiniteDifferenceMethod
    grid::NTuple{P,Float64}
    coefs::NTuple{P,Float64}
    condition::Float64
    factor::Float64
    bound_estimator::E
end

"""
    FiniteDifferenceMethod(
        grid::AbstractVector,
        q::Int;
        condition::Real=DEFAULT_CONDITION,
        factor::Real=DEFAULT_FACTOR
    )

Construct a finite difference method.

# Arguments
- `grid::Abstract`: The grid. See [`FiniteDifferenceMethod`](@ref).
- `q::Int`: Order of the derivative to estimate.

# Keyword Arguments
- `condition::Real`: Condition number. See [`DEFAULT_CONDITION`](@ref).
- `factor::Real`: Factor number. See [`DEFAULT_FACTOR`](@ref).

# Returns
- `FiniteDifferenceMethod`: Specified finite difference method.
"""
function FiniteDifferenceMethod(
    grid::AbstractVector,
    q::Int;
    condition::Real=DEFAULT_CONDITION,
    factor::Real=DEFAULT_FACTOR
)
    p = length(grid)
    _check_p_q(p, q)
    return UnadaptedFiniteDifferenceMethod{p,q}(
        grid,
        _coefs(grid, q),
        Float64(condition),
        Float64(factor)
    )
end

"""
    (m::FiniteDifferenceMethod)(
        f::Function,
        x::T;
        factor::Real=1,
        max_step::Real=0.1 * max(abs(x), one(x))
    ) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and an
automatically determined step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.

# Keywords
- `factor::Real=1`: Factor to amplify the estimated round-off error by. This can be used
    to force a more conservative step size.
- `max_step::Real=0.1 * max(abs(x), one(x))`: Maximum step size.

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
@inline function (m::FiniteDifferenceMethod)(f::Function, x::Real; kw_args...)
    # Assume that converting to float is desired.
    return _call_method(m, f, float(x); kw_args...)
end
@inline function _call_method(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    factor::Real=1,
    max_step::Real=0.1 * max(abs(x), one(x))
) where T<:AbstractFloat
    # The automatic step size calculation fails if `m.q == 0`, so handle that edge case.
    iszero(m.q) && return f(x)
    h, _ = estimate_step(m, f, x, factor=factor, max_step=max_step)
    return _eval_method(m, f, x, h)
end

"""
    (m::FiniteDifferenceMethod)(f::Function, x::T, h::Real) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and a given
step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.
- `h::Real`: Step size.

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
@inline function (m::FiniteDifferenceMethod)(f::Function, x::Real, h::Real)
    # Assume that converting to float is desired.
    return _eval_method(m, f, float(x), h)
end
@inline function _eval_method(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T,
    h::Real
) where T<:AbstractFloat
    return sum(
        i -> convert(T, m.coefs[i]) * f(T(x + h * m.grid[i])),
        eachindex(m.grid)
    ) / h^m.q
end

# Check the method and derivative orders for consistency.
function _check_p_q(p::Integer, q::Integer)
    q >= 0 || throw(DomainError(q, "order of derivative (`q`) must be non-negative"))
    q < p || throw(DomainError(
        (q, p),
        "order of the method (q) must be strictly greater than that of the derivative (p)",
    ))
    # Check whether the method can be computed. We require the factorial of the method order
    # to be computable with regular `Int`s, but `factorial` will after 20, so 20 is the
    # largest we can allow.
    p > 20 && throw(DomainError(p, "order of the method (`p`) is too large to be computed"))
    return
end

const _COEFFS_CACHE = Dict{Tuple{Tuple{Vararg{Int}}, Integer}, Tuple{Vararg{Float64}}}()

# Compute coefficients for the method and cache the result.
function _coefs(grid::Tuple{Vararg{Int}}, q::Integer) where N
    return get!(_COEFFS_CACHE, (grid, q)) do
        p = length(grid)
        # For high precision on the `\`, we use `Rational`, and to prevent overflows we use
        # `Int128`. At the end we go to `Float64` for fast floating point math, rather than
        # rational math.
        C = [Rational{Int128}(g^i) for i in 0:(p - 1), g in grid]
        x = zeros(Rational{Int128}, p)
        x[q + 1] = factorial(q)
        return Tuple(Float64.(C \ x))
    end
end

# Estimate the bound on the derivative by amplifying the ∞-norm.
function _make_default_bound_estimator(; condition::Real=DEFAULT_CONDITION)
    default_bound_estimator(f, x) = condition * estimate_magitude(f, x)
    return default_bound_estimator
end

function Base.show(io::IO, m::MIME"text/plain", x::FiniteDifferenceMethod)
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
        factor::Real=1,
        max_step::Real=0.1 * max(abs(x), one(x))
    ) where T<:AbstractFloat

Estimate the step size for a finite difference method `m`. Also estimates the error of the
estimate of the derivative.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.

# Keywords
- `factor::Real=1`. Factor to amplify the estimated round-off error by. This can be used
    to force a more conservative step size.
- `max_step::Real=0.1 * max(abs(x), one(x))`: Maximum step size.

# Returns
- `Tuple{T, <:AbstractFloat}`: Estimated step size and an estimate of the error of the
    finite difference estimate.
"""
function estimate_step(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    factor::Real=1,
    max_step::Real=0.1 * max(abs(x), one(x))
) where T<:AbstractFloat
    p = length(m.coefs)
    q = m.q

    # Estimate the round-off error.
    ε = estimate_roundoff_error(f, x) * factor

    # Estimate the bound on the derivatives.
    M = m.bound_estimator(f, x)

    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = ε * sum(abs, m.coefs)
    C₂ = M * sum(n -> abs(m.coefs[n] * m.grid[n]^p), eachindex(m.coefs)) / factorial(p)
    # Type inference fails on this, so we annotate it, which gives big performance benefits.
    h::T = convert(T, min((q / (p - q) * (C₁ / C₂))^(1 / p), max_step))

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
            condition::Real=DEFAULT_CONDITION,
            factor::Real=DEFAULT_FACTOR,
            geom::Bool=false
        )
            _check_p_q(p, q)
            grid = $grid_fun(p)
            geom && (grid = _exponentiate_grid(grid))
            coefs = _coefs(grid, q)
            if adapt >= 1
                bound_estimator = $fdm_fun(
                    p + 2
                    p;
                    adapt=adapt - 1,
                    condition=condition,
                    factor=factor
                )
                return AdaptedFiniteDifferenceMethod{p, q, typeof(bound_estimator)}(
                    grid,
                    coefs,
                    condition,
                    factor,
                    bound_estimator
                )
            else
                return UnadaptedFiniteDifferenceMethod{p, q}(grid, coefs, condition, factor)
            end
        end

        @doc """
    $($(Meta.quot(fdm_fun)))(
        p::Int,
        q::Int;
        adapt::Int=1,
        condition::Real=DEFAULT_CONDITION,
        factor::Real=DEFAULT_FACTOR,
        geom::Bool=false
    )

Contruct a finite difference method at a $($(Meta.quot(direction))) grid of `p` points.

# Arguments
- `p::Int`: Number of grid points.
- `q::Int`: Order of the derivative to estimate.

# Keywords
- `adapt::Int=1`: Use another finite difference method to estimate the magnitude of the
    `p`th order derivative, which is important for the step size computation. Recurse
    this procedure `adapt` times.
- `condition::Real`: Condition number. See [`DEFAULT_CONDITION`](@ref).
- `factor::Real`: Factor number. See [`DEFAULT_FACTOR`](@ref).
- `geom::Bool`: Use geometrically spaced points instead of linearly spaced points.

# Returns
- `FiniteDifferenceMethod`: The specified finite difference method.
        """ $fdm_fun
    end
end

_forward_grid(p::Int) = Tuple(0:(p - 1))

_backward_grid(p::Int) = Tuple((1 - p):0)

function _central_grid(p::Int)
    if isodd(p)
        return Tuple(div(1 - p, 2):div(p - 1, 2))
    else
        return ((div(-p, 2):-1)..., (1:div(p, 2))...)
    end
end

function _exponentiate_grid(grid::Tuple{Vararg{Int}}, base::Int=3)
    return sign.(grid) .* div.(base .^ abs.(grid), base)
end

function _is_symmetric(vec::Vector; centre_zero::Bool=false, negate_half::Bool=false)
    half_sign = negate_half ? -1 : 1
    if isodd(length(vec))
        centre_zero && vec[end ÷ 2 + 1] != 0 && return false
        return vec[1:end ÷ 2] == half_sign .* reverse(vec[(end ÷ 2 + 2):end])
    else
        return vec[1:end ÷ 2] == half_sign .* reverse(vec[(end ÷ 2 + 1):end])
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
        x::T,
        h::Real=0.1 * max(abs(x), one(x));
        power=nothing,
        breaktol=Inf,
        kw_args...
    ) where T<:AbstractFloat

Use Richardson extrapolation to refine a finite difference method.

Takes further in keyword arguments for `Richardson.extrapolate`. This method
automatically sets `power = 2` if `m` is symmetric and `power = 1`. Moreover, it defaults
`breaktol = Inf`.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.
- `h::Real=0.1 * max(abs(x), one(x))`: Initial step size.

# Returns
- `Tuple{<:AbstractFloat, <:AbstractFloat}`: Estimate of the derivative and error.
"""
function extrapolate_fdm(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T,
    h::Real=0.1 * max(abs(x), one(x));
    power::Int=1,
    breaktol::Real=Inf,
    kw_args...
) where T<:AbstractFloat
    (power == 1 && _is_symmetric(m)) && (power = 2)
    return extrapolate(h -> m(f, x, h), h; power=power, breaktol=breaktol, kw_args...)
end
