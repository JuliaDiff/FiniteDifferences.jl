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

abstract type FiniteDifferenceMethod{P,Q} end

"""
    UnadaptedFiniteDifferenceMethod{P,Q} <: FiniteDifferenceMethod{P,Q}

A finite difference method that estimates a `Q`th order derivative from `P` function
evaluations. This method does not dynamically adapt its step size.

# Fields
- `grid::NTuple{P,Float64}`: Multiples of the step size that the function will be evaluated
    at.
- `coefs::NTuple{P,Float64}`: Coefficients corresponding to the grid functions that the
    function evaluations will be weighted by.
- `coefs_neighbourhood::NTuple{3,NTuple{P,Float64}}`: Sets of coefficients used for
    estimating the magnitude of the derivative in a neighbourhood.
- `bound_estimator::FiniteDifferenceMethod`: A finite difference method that is tuned to
    perform adaptation for this finite difference method.
- `condition::Float64`: Condition number. See See [`DEFAULT_CONDITION`](@ref).
- `factor::Float64`: Factor number. See See [`DEFAULT_FACTOR`](@ref).
- `∇f_magnitude_mult::Float64`: Internally computed quantity.
- `f_error_mult::Float64`: Internally computed quantity.
"""
struct UnadaptedFiniteDifferenceMethod{P,Q} <: FiniteDifferenceMethod{P,Q}
    grid::NTuple{P,Int}
    coefs::NTuple{P,Float64}
    coefs_neighbourhood::NTuple{3,NTuple{P,Float64}}
    condition::Float64
    factor::Float64
    ∇f_magnitude_mult::Float64
    f_error_mult::Float64
end

"""
    AdaptedFiniteDifferenceMethod{
        P,
        Q,
        E<:FiniteDifferenceMethod
    } <: FiniteDifferenceMethod{P,Q}

A finite difference method that estimates a `Q`th order derivative from `P` function
evaluations. This method does dynamically adapt its step size.

# Fields
- `grid::NTuple{P,Float64}`: Multiples of the step size that the function will be evaluated
    at.
- `coefs::NTuple{P,Float64}`: Coefficients corresponding to the grid functions that the
    function evaluations will be weighted by.
- `coefs_neighbourhood::NTuple{3,NTuple{P,Float64}}`: Sets of coefficients used for
    estimating the magnitude of the derivative in a neighbourhood.
- `condition::Float64`: Condition number. See See [`DEFAULT_CONDITION`](@ref).
- `factor::Float64`: Factor number. See See [`DEFAULT_FACTOR`](@ref).
- `∇f_magnitude_mult::Float64`: Internally computed quantity.
- `f_error_mult::Float64`: Internally computed quantity.
- `bound_estimator::FiniteDifferenceMethod`: A finite difference method that is tuned to
    perform adaptation for this finite difference method.
"""
struct AdaptedFiniteDifferenceMethod{
    P,
    Q,
    E<:FiniteDifferenceMethod
} <: FiniteDifferenceMethod{P,Q}
    grid::NTuple{P,Int}
    coefs::NTuple{P,Float64}
    coefs_neighbourhood::NTuple{3,NTuple{P,Float64}}
    condition::Float64
    factor::Float64
    ∇f_magnitude_mult::Float64
    f_error_mult::Float64
    bound_estimator::E
end

"""
    FiniteDifferenceMethod(
        grid::NTuple{P,Int},
        q::Int;
        condition::Real=DEFAULT_CONDITION,
        factor::Real=DEFAULT_FACTOR
    )

Construct a finite difference method.

# Arguments
- `grid::NTuple{P,Int}`: The grid. See [`AdaptedFiniteDifferenceMethod`](@ref) or
    [`UnadaptedFiniteDifferenceMethod`](@ref).
- `q::Int`: Order of the derivative to estimate.

# Keywords
- `condition::Real`: Condition number. See [`DEFAULT_CONDITION`](@ref).
- `factor::Real`: Factor number. See [`DEFAULT_FACTOR`](@ref).

# Returns
- `FiniteDifferenceMethod`: Specified finite difference method.
"""
function FiniteDifferenceMethod(
    grid::NTuple{P,Int},
    q::Int;
    condition::Real=DEFAULT_CONDITION,
    factor::Real=DEFAULT_FACTOR
) where P
    _check_p_q(P, q)
    coefs, coefs_neighbourhood, ∇f_magnitude_mult, f_error_mult = _coefs(grid, q)
    return UnadaptedFiniteDifferenceMethod{P,q}(
        grid,
        coefs,
        coefs_neighbourhood,
        Float64(condition),
        Float64(factor),
        ∇f_magnitude_mult,
        f_error_mult
    )
end

"""
    (m::FiniteDifferenceMethod)(
        f::Function,
        x::T;
        max_range::Real=Inf
    ) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and an
automatically determined step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.

# Keywords
- `max_range::Real=Inf`: Upper bound on how far `f` can be evaluated away from `x`.

# Returns
- Estimate of the derivative.

# Examples

```julia-repl
julia> fdm = central_fdm(5, 1)
FiniteDifferenceMethod:
  order of method:       5
  order of derivative:   1
  grid:                  (-2, -1, 0, 1, 2)
  coefficients:          (0.08333333333333333, -0.6666666666666666, 0.0, 0.6666666666666666, -0.08333333333333333)

julia> fdm(sin, 1)
0.5403023058681607

julia> fdm(sin, 1) - cos(1)  # Check the error.
2.098321516541546e-14

julia> FiniteDifferences.estimate_step(fdm, sin, 1.0)  # Computes step size and estimates the error.
(0.001065235154086019, 1.9541865128909085e-13)
```
"""
@inline function (m::FiniteDifferenceMethod)(f::Function, x::Real; kw_args...)
    # Assume that converting to float is desired.
    return _call_method(m, f, float(x); kw_args...)
end
@inline function _call_method(
    m::FiniteDifferenceMethod{P,0},
    f::Function,
    x::T;
    max_range::Real=Inf
) where {P,T<:AbstractFloat}
    # The automatic step size calculation fails if `Q == 0`, so handle that edge case.
     return f(x)
 end
@inline function _call_method(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    max_range::Real=Inf
) where {T<:AbstractFloat}
    step, _ = estimate_step(m, f, x, max_range=max_range)
    return _eval_method(m, _evals(m, f, x, step), x, step, m.coefs)
end

function _estimate_magnitudes(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T;
    max_range::Real=Inf
) where {T<:AbstractFloat}
    step, _ = estimate_step(m, f, x, max_range=max_range)
    fs = _evals(m, f, x, step)
    # Estimate magnitude of `∇f` in a neighbourhood of `x`.
    ∇fs = _eval_method.((m,), (fs,), x, step, m.coefs_neighbourhood)
    ∇f_magnitude = maximum(maximum.(abs.(∇fs)))
    # Estimate magnitude of `f` in a neighbourhood of `x`.
    f_magnitude = maximum(maximum.(abs.(fs)))
    return ∇f_magnitude, f_magnitude
end

"""
    (m::FiniteDifferenceMethod)(f::Function, x::T, step::Real) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and a given
step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.
- `step::Real`: Step size.

# Returns
- Estimate of the derivative.

# Examples

```julia-repl
julia> fdm = central_fdm(5, 1)
FiniteDifferenceMethod:
  order of method:       5
  order of derivative:   1
  grid:                  (-2, -1, 0, 1, 2)
  coefficients:          (0.08333333333333333, -0.6666666666666666, 0.0, 0.6666666666666666, -0.08333333333333333)

 julia> fdm(sin, 1, 1e-3)
 0.5403023058679624

julia> fdm(sin, 1, 1e-3) - cos(1)  # Check the error.
-1.7741363933510002e-13
```
"""
@inline function (m::FiniteDifferenceMethod)(f::Function, x::Real, step::Real)
    # Assume that converting to float is desired.
    x = float(x)
    return _eval_method(m, _evals(m, f, x, step), x, step, m.coefs)
end
@inline function _evals(
    m::FiniteDifferenceMethod,
    f::Function,
    x::T,
    step::T
) where {T<:AbstractFloat}
    return f.(x .+ step .* m.grid)
end
@inline function _eval_method(
    m::FiniteDifferenceMethod{P,Q},
    fs::NTuple{P},
    x::T,
    step::Real,
    coefs::NTuple{P,Float64}
) where {P,Q,T<:AbstractFloat}
    return sum(T.(coefs) .* fs) ./ T(step^Q)
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

function _compute_coefs(grid, p, q)
    # For high precision on the `\`, we use `Rational`, and to prevent overflows we use
    # `Int128`. At the end we go to `Float64` for fast floating point math, rather than
    # rational math.
    C = [Rational{Int128}(g^i) for i in 0:(p - 1), g in grid]
    x = zeros(Rational{Int128}, p)
    x[q + 1] = factorial(q)
    return Tuple(Float64.(C \ x))
end

const _COEFFS_CACHE = Dict{
    Tuple{Tuple{Vararg{Int}},Integer},
    Tuple{Tuple{Vararg{Float64}},Tuple{Vararg{Tuple{Vararg{Float64}}}},Float64,Float64}
}()

# Compute coefficients for the method and cache the result.
function _coefs(grid::NTuple{P, Int}, q::Integer) where P
    return get!(_COEFFS_CACHE, (grid, q)) do
        coefs = _compute_coefs(grid, P, q)
        # Compute coefficients for a neighbourhood estimate.
        if all(grid .>= 0)
            coefs_neighbourhood = (
                _compute_coefs(grid .- 2, P, q),
                _compute_coefs(grid .- 1, P, q),
                _compute_coefs(grid, P, q)
            )
        elseif all(grid .<= 0)
            coefs_neighbourhood = (
                _compute_coefs(grid, P, q),
                _compute_coefs(grid .+ 1, P, q),
                _compute_coefs(grid .+ 2, P, q)
            )
        else
            coefs_neighbourhood = (
                _compute_coefs(grid .- 1, P, q),
                _compute_coefs(grid, P, q),
                _compute_coefs(grid .+ 1, P, q)
            )
        end
        # Ccompute multipliers.
        ∇f_magnitude_mult = sum(abs.(coefs .* grid .^ P)) / factorial(P)
        f_error_mult = sum(abs.(coefs))
        return coefs, coefs_neighbourhood, ∇f_magnitude_mult, f_error_mult
    end
end

function Base.show(
    io::IO,
    m::MIME"text/plain",
    x::FiniteDifferenceMethod{P, Q}
) where {P, Q}
    @printf io "FiniteDifferenceMethod:\n"
    @printf io "  order of method:       %d\n" P
    @printf io "  order of derivative:   %d\n" Q
    @printf io "  grid:                  %s\n" x.grid
    @printf io "  coefficients:          %s\n" x.coefs
end

"""
    function estimate_step(
        m::FiniteDifferenceMethod,
        f::Function,
        x::T;
        max_range::Real=Inf
    ) where T<:AbstractFloat

Estimate the step size for a finite difference method `m`. Also estimates the error of the
estimate of the derivative.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.

# Keywords
- `max_range::Real=Inf`: Upper bound on how far `f` can be evaluated away from `x`.

# Returns
- `Tuple{<:AbstractFloat, <:AbstractFloat}`: Estimated step size and an estimate of the
    error of the finite difference estimate.
"""
function estimate_step(
    m::UnadaptedFiniteDifferenceMethod,
    f::Function,
    x::T;
    max_range::Real=Inf
) where {T<:AbstractFloat}
    step, acc = _estimate_step_acc(m, x, max_range)
    return step, acc
end
function estimate_step(
    m::AdaptedFiniteDifferenceMethod,
    f::Function,
    x::T;
    max_range::Real=Inf
) where {T<:AbstractFloat}
    ∇f_magnitude, f_magnitude = _estimate_magnitudes(
        m.bound_estimator,
        f,
        x;
        max_range=max_range
    )
    if ∇f_magnitude == 0 || f_magnitude == 0
        step, acc = _estimate_step_acc(m, x, max_range)
    else
        step, acc = _estimate_step_acc(m, x, ∇f_magnitude, eps(f_magnitude), max_range)
    end
    return step, acc
end

function _compute_step_acc(
    m::FiniteDifferenceMethod{P,Q},
    ∇f_magnitude::Real,
    f_error::Real
) where {P,Q}
    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = f_error * m.f_error_mult * m.factor
    C₂ = ∇f_magnitude * m.∇f_magnitude_mult
    step = (Q / (P - Q) * (C₁ / C₂))^(1 / P)
    # Estimate the accuracy of the method.
    acc = C₁ * step^(-Q) + C₂ * step^(P - Q)
    return step, acc
end

function _compute_default(
    m::FiniteDifferenceMethod,
    x::T
) where {T<:AbstractFloat}
    return _compute_step_acc(m, m.condition, eps(T))
end

function _estimate_step_acc(
    m::FiniteDifferenceMethod,
    x::T,
    max_range::Real
) where {T<:AbstractFloat}
    step, acc = _compute_default(m, x)
    return _limit_step(m, x, step, acc, max_range)
end
function _estimate_step_acc(
    m::AdaptedFiniteDifferenceMethod{P,Q,E},
    x::T,
    ∇f_magnitude::Real,
    f_error::Real,
    max_range::Real
) where {P,Q,E,T<:AbstractFloat}
    step, acc = _compute_step_acc(m, ∇f_magnitude, f_error)
    return _limit_step(m, x, step, acc, max_range)
end

function _limit_step(
    m::FiniteDifferenceMethod,
    x::T,
    step::Real,
    acc::Real,
    max_range::Real
) where {T<:AbstractFloat}
    # First, limit the step size based on the maximum range.
    step_max = max_range / maximum(abs.(m.grid))
    if step > step_max
        step = step_max
        acc = NaN
    end
    # Second, prevent very large step sizes, which can occur for high-order methods or
    # slowly-varying functions.
    step_default, _ = _compute_default(m, x)
    step_max_default = 1000step_default
    if step > step_max_default
        step = step_max_default
        acc = NaN
    end
    return step, acc
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
            coefs, coefs_neighbourhood, ∇f_magnitude_mult, f_error_mult = _coefs(grid, q)
            if adapt >= 1
                bound_estimator = $fdm_fun(
                    # We need to increase the order by two to be able to estimate the
                    # magnitude of the derivative of `f` in a neighbourhood of `x`.
                    p + 2,
                    p;
                    adapt=adapt - 1,
                    condition=condition,
                    factor=factor,
                    geom=geom
                )
                return AdaptedFiniteDifferenceMethod{p, q, typeof(bound_estimator)}(
                    grid,
                    coefs,
                    coefs_neighbourhood,
                    Float64(condition),
                    Float64(factor),
                    ∇f_magnitude_mult,
                    f_error_mult,
                    bound_estimator
                )
            else
                return UnadaptedFiniteDifferenceMethod{p, q}(
                    grid,
                    coefs,
                    coefs_neighbourhood,
                    Float64(condition),
                    Float64(factor),
                    ∇f_magnitude_mult,
                    f_error_mult
                )
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

function _is_symmetric(vec::Tuple; centre_zero::Bool=false, negate_half::Bool=false)
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
        x::Real,
        initial_step::Real=10,
        power::Int=1,
        breaktol::Real=Inf,
        kw_args...
    ) where T<:AbstractFloat

Use Richardson extrapolation to extrapolate a finite difference method.

Takes further in keyword arguments for `Richardson.extrapolate`. This method
automatically sets `power = 2` if `m` is symmetric and `power = 1`. Moreover, it defaults
`breaktol = Inf`.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::Real`: Point to estimate the derivative at.
- `initial_step::Real=10`: Initial step size.

# Returns
- `Tuple{<:AbstractFloat, <:AbstractFloat}`: Estimate of the derivative and error.
"""
function extrapolate_fdm(
    m::FiniteDifferenceMethod,
    f::Function,
    x::Real,
    initial_step::Real=10,
    power::Int=1,
    breaktol::Real=Inf,
    kw_args...
) where T<:AbstractFloat
    (power == 1 && _is_symmetric(m)) && (power = 2)
    return extrapolate(
        step -> m(f, x, step),
        float(initial_step);
        power=power,
        breaktol=breaktol,
        kw_args...
    )
end
