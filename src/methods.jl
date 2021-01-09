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
- `grid::SVector{P,Int}`: Multiples of the step size that the function will be evaluated at.
- `coefs::SVector{P,Float64}`: Coefficients corresponding to the grid functions that the
    function evaluations will be weighted by.
- `coefs_neighbourhood::NTuple{3,SVector{P,Float64}}`: Sets of coefficients used for
    estimating the magnitude of the derivative in a neighbourhood.
- `bound_estimator::FiniteDifferenceMethod`: A finite difference method that is tuned to
    perform adaptation for this finite difference method.
- `condition::Float64`: Condition number. See See [`DEFAULT_CONDITION`](@ref).
- `factor::Float64`: Factor number. See See [`DEFAULT_FACTOR`](@ref).
- `max_range::Float64`: Maximum distance that a function is evaluated from the input at
    which the derivative is estimated.
- `∇f_magnitude_mult::Float64`: Internally computed quantity.
- `f_error_mult::Float64`: Internally computed quantity.
"""
struct UnadaptedFiniteDifferenceMethod{P,Q} <: FiniteDifferenceMethod{P,Q}
    grid::SVector{P,Int}
    coefs::SVector{P,Float64}
    coefs_neighbourhood::NTuple{3,SVector{P,Float64}}
    condition::Float64
    factor::Float64
    max_range::Float64
    ∇f_magnitude_mult::Float64
    f_error_mult::Float64
end

"""
    AdaptedFiniteDifferenceMethod{
        P, Q, E<:FiniteDifferenceMethod
    } <: FiniteDifferenceMethod{P,Q}

A finite difference method that estimates a `Q`th order derivative from `P` function
evaluations.

This method dynamically adapts its step size. The adaptation works by explicitly estimating
the truncation error and round-off error, and choosing the step size to optimally balance
those. The truncation error is given by the magnitude of the `P`th order derivative, which
will be estimated with another finite difference method (`E`). This finite difference
method, `bound_estimator`, will be tasked with estimating the `P`th order derivative in a
_neighbourhood_, not just at some `x`. To do this, it will use a careful reweighting of the
function evaluations to estimate the `P`th order derivative at, in the case of a central
method, `x - h`, `x`, and `x + h`, where `h` is the step size. The coeffients for this
estimate, the _neighbourhood estimate_, are given by the three sets of coeffients in
`bound_estimator.coefs_neighbourhood`. The round-off error is estimated by the round-off
error of the function evaluations performed by `bound_estimator`. The trunction error is
amplified by `condition`, and the round-off error is amplified by `factor`. The quantities
`∇f_magnitude_mult` and `f_error_mult` are precomputed quantities that facilitate the
step size adaptation procedure.

# Fields
- `grid::SVector{P,Int}`: Multiples of the step size that the function will be evaluated at.
- `coefs::SVector{P,Float64}`: Coefficients corresponding to the grid functions that the
    function evaluations will be weighted by.
- `coefs_neighbourhood::NTuple{3,SVector{P,Float64}}`: Sets of coefficients used for
    estimating the magnitude of the derivative in a neighbourhood.
- `condition::Float64`: Condition number. See See [`DEFAULT_CONDITION`](@ref).
- `factor::Float64`: Factor number. See See [`DEFAULT_FACTOR`](@ref).
- `max_range::Float64`: Maximum distance that a function is evaluated from the input at
    which the derivative is estimated.
- `∇f_magnitude_mult::Float64`: Internally computed quantity.
- `f_error_mult::Float64`: Internally computed quantity.
- `bound_estimator::FiniteDifferenceMethod`: A finite difference method that is tuned to
    perform adaptation for this finite difference method.
"""
struct AdaptedFiniteDifferenceMethod{
    P, Q, E<:FiniteDifferenceMethod
} <: FiniteDifferenceMethod{P,Q}
    grid::SVector{P,Int}
    coefs::SVector{P,Float64}
    coefs_neighbourhood::NTuple{3,SVector{P,Float64}}
    condition::Float64
    factor::Float64
    max_range::Float64
    ∇f_magnitude_mult::Float64
    f_error_mult::Float64
    bound_estimator::E
end

"""
    FiniteDifferenceMethod(
        grid::AbstractVector{Int},
        q::Int;
        condition::Real=DEFAULT_CONDITION,
        factor::Real=DEFAULT_FACTOR,
        max_range::Real=Inf
    )

Construct a finite difference method.

# Arguments
- `grid::Vector{Int}`: The grid. See [`AdaptedFiniteDifferenceMethod`](@ref) or
    [`UnadaptedFiniteDifferenceMethod`](@ref).
- `q::Int`: Order of the derivative to estimate.

# Keywords
- `condition::Real`: Condition number. See [`DEFAULT_CONDITION`](@ref).
- `factor::Real`: Factor number. See [`DEFAULT_FACTOR`](@ref).
- `max_range::Real=Inf`: Maximum distance that a function is evaluated from the input at
    which the derivative is estimated.

# Returns
- `FiniteDifferenceMethod`: Specified finite difference method.
"""
function FiniteDifferenceMethod(
    grid::SVector{P,Int},
    q::Int;
    condition::Real=DEFAULT_CONDITION,
    factor::Real=DEFAULT_FACTOR,
    max_range::Real=Inf,
) where P
    _check_p_q(P, q)
    coefs, coefs_neighbourhood, ∇f_magnitude_mult, f_error_mult = _coefs_mults(grid, q)
    return UnadaptedFiniteDifferenceMethod{P,q}(
        grid,
        coefs,
        coefs_neighbourhood,
        condition,
        factor,
        max_range,
        ∇f_magnitude_mult,
        f_error_mult
    )
end
function FiniteDifferenceMethod(grid::AbstractVector{Int}, q::Int; kw_args...)
    return FiniteDifferenceMethod(SVector{length(grid)}(grid), q; kw_args...)
end

"""
    (m::FiniteDifferenceMethod)(f::Function, x::T) where T<:AbstractFloat

Estimate the derivative of `f` at `x` using the finite differencing method `m` and an
automatically determined step size.

# Arguments
- `f::Function`: Function to estimate derivative of.
- `x::T`: Input to estimate derivative at.

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
0.5403023058681607

julia> fdm(sin, 1) - cos(1)  # Check the error.
2.098321516541546e-14

julia> FiniteDifferences.estimate_step(fdm, sin, 1.0)  # Computes step size and estimates the error.
(0.001065235154086019, 1.9541865128909085e-13)
```
"""
# We loop over all concrete subtypes of `FiniteDifferenceMethod` for Julia v1.0 compatibility.
for T in (UnadaptedFiniteDifferenceMethod, AdaptedFiniteDifferenceMethod)
    @eval begin
        function (m::$T)(f::TF, x::Real) where TF<:Function
            x = float(x)  # Assume that converting to float is desired, if it isn't already.
            step = first(estimate_step(m, f, x))
            return m(f, x, step)
        end
        function (m::$T{P,0})(f::TF, x::Real) where {P,TF<:Function}
            # The automatic step size calculation fails if `Q == 0`, so handle that edge
            # case.
            return f(x)
        end
    end
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
  grid:                  [-2, -1, 0, 1, 2]
  coefficients:          [0.08333333333333333, -0.6666666666666666, 0.0, 0.6666666666666666, -0.08333333333333333]

julia> fdm(sin, 1, 1e-3)
 0.5403023058679624

julia> fdm(sin, 1, 1e-3) - cos(1)  # Check the error.
-1.7741363933510002e-13
```
"""
# We loop over all concrete subtypes of `FiniteDifferenceMethod` for 1.0 compatibility.
for T in (UnadaptedFiniteDifferenceMethod, AdaptedFiniteDifferenceMethod)
    @eval begin
        function (m::$T{P,Q})(f::TF, x::Real, step::Real) where {P,Q,TF<:Function}
            x = float(x)  # Assume that converting to float is desired, if it isn't already.
            fs = _eval_function(m, f, x, step)
            return _compute_estimate(m, fs, x, step, m.coefs)
        end
    end
end

function _eval_function(
    m::FiniteDifferenceMethod, f::TF, x::T, step::Real,
) where {TF<:Function,T<:AbstractFloat}
    return f.(x .+ T(step) .* m.grid)
end

function _compute_estimate(
    m::FiniteDifferenceMethod{P,Q},
    fs::SVector{P,TF},
    x::T,
    step::Real,
    coefs::SVector{P,Float64},
) where {P,Q,TF,T<:AbstractFloat}
    # If we substitute `T.(coefs)` in the expression below, then allocations occur. We
    # therefore perform the broadcasting first. See
    # https://github.com/JuliaLang/julia/issues/39151.
    coefs = T.(coefs)
    return sum(fs .* coefs) ./ T(step)^Q
end

# Check the method and derivative orders for consistency.
function _check_p_q(p::Integer, q::Integer)
    q >= 0 || throw(DomainError(q, "order of derivative (`q`) must be non-negative"))
    q < p || throw(DomainError(
        (q, p),
        "order of the method (`p`) must be strictly greater than that of the derivative " *
        "(`q`)",
    ))
end

function _coefs(grid, p, q)
    # For high precision on the `\`, we use `Rational`, and to prevent overflows we use
    # `BigInt`. At the end we go to `Float64` for fast floating point math, rather than
    # rational math.
    C = [Rational{BigInt}(g^i) for i in 0:(p - 1), g in grid]
    x = zeros(Rational{BigInt}, p)
    x[q + 1] = factorial(big(q))
    return SVector{p}(Float64.(C \ x))
end

const _COEFFS_MULTS_CACHE = Dict{
    Tuple{SVector,Integer},  # Keys: (grid, q)
    # Values: (coefs, coefs_neighbourhood, ∇f_magnitude_mult, f_error_mult)
    Tuple{SVector,Tuple{Vararg{SVector}},Float64,Float64}
}()

# Compute coefficients for the method and cache the result.
function _coefs_mults(grid::SVector{P, Int}, q::Integer) where P
    return get!(_COEFFS_MULTS_CACHE, (grid, q)) do
        # Compute coefficients for derivative estimate.
        coefs = _coefs(grid, P, q)
        # Compute coefficients for a neighbourhood estimate.
        if all(grid .>= 0)
            coefs_neighbourhood = (
                _coefs(grid .- 2, P, q),
                _coefs(grid .- 1, P, q),
                _coefs(grid, P, q)
            )
        elseif all(grid .<= 0)
            coefs_neighbourhood = (
                _coefs(grid, P, q),
                _coefs(grid .+ 1, P, q),
                _coefs(grid .+ 2, P, q)
            )
        else
            coefs_neighbourhood = (
                _coefs(grid .- 1, P, q),
                _coefs(grid, P, q),
                _coefs(grid .+ 1, P, q)
            )
        end
        # Compute multipliers.
        ∇f_magnitude_mult = sum(abs.(coefs .* grid .^ P)) / factorial(big(P))
        f_error_mult = sum(abs.(coefs))
        return coefs, coefs_neighbourhood, ∇f_magnitude_mult, f_error_mult
    end
end

function Base.show(
    io::IO,
    m::MIME"text/plain",
    x::FiniteDifferenceMethod{P, Q},
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
        x::T
    ) where T<:AbstractFloat

Estimate the step size for a finite difference method `m`. Also estimates the error of the
estimate of the derivative.

# Arguments
- `m::FiniteDifferenceMethod`: Finite difference method to estimate the step size for.
- `f::Function`: Function to evaluate the derivative of.
- `x::T`: Point to estimate the derivative at.

# Returns
- `Tuple{<:AbstractFloat, <:AbstractFloat}`: Estimated step size and an estimate of the
    error of the finite difference estimate. The error will be `NaN` if the method failed
    to estimate the error.
"""
function estimate_step(
    m::UnadaptedFiniteDifferenceMethod, f::TF, x::T,
) where {TF<:Function,T<:AbstractFloat}
    step, acc = _compute_step_acc_default(m, x)
    return _limit_step(m, x, step, acc)
end
function estimate_step(
    m::AdaptedFiniteDifferenceMethod{P,Q}, f::TF, x::T,
) where {P,Q,TF<:Function,T<:AbstractFloat}
    ∇f_magnitude, f_magnitude = _estimate_magnitudes(m.bound_estimator, f, x)
    if ∇f_magnitude == 0.0 || f_magnitude == 0.0
        step, acc = _compute_step_acc_default(m, x)
    else
        step, acc = _compute_step_acc(m, ∇f_magnitude, eps(f_magnitude))
    end
    return _limit_step(m, x, step, acc)
end

function _estimate_magnitudes(
    m::FiniteDifferenceMethod{P,Q}, f::TF, x::T,
) where {P,Q,TF<:Function,T<:AbstractFloat}
    step = first(estimate_step(m, f, x))
    fs = _eval_function(m, f, x, step)
    # Estimate magnitude of `∇f` in a neighbourhood of `x`.
    ∇fs = SVector{3}(
        _compute_estimate(m, fs, x, step, m.coefs_neighbourhood[1]),
        _compute_estimate(m, fs, x, step, m.coefs_neighbourhood[2]),
        _compute_estimate(m, fs, x, step, m.coefs_neighbourhood[3])
    )
    ∇f_magnitude = maximum(maximum.(abs, ∇fs))
    # Estimate magnitude of `f` in a neighbourhood of `x`.
    f_magnitude = maximum(maximum.(abs, fs))
    return ∇f_magnitude, f_magnitude
end

function _compute_step_acc_default(m::FiniteDifferenceMethod, x::T) where {T<:AbstractFloat}
    # Compute a default step size using a heuristic and [`DEFAULT_CONDITION`](@ref).
    return _compute_step_acc(m, m.condition, eps(T))
end

function _compute_step_acc(
    m::FiniteDifferenceMethod{P,Q}, ∇f_magnitude::Real, f_error::Real,
) where {P,Q}
    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = f_error * m.f_error_mult * m.factor
    C₂ = ∇f_magnitude * m.∇f_magnitude_mult
    step = (Q / (P - Q) * (C₁ / C₂))^(1 / P)
    # Estimate the accuracy of the method.
    acc = C₁ * step^(-Q) + C₂ * step^(P - Q)
    return step, acc
end

function _limit_step(
    m::FiniteDifferenceMethod, x::T, step::Real, acc::Real,
) where {T<:AbstractFloat}
    # First, limit the step size based on the maximum range.
    step_max = m.max_range / maximum(abs.(m.grid))
    if step > step_max
        step = step_max
        acc = NaN
    end
    # Second, prevent very large step sizes, which can occur for high-order methods or
    # slowly-varying functions.
    step_default, _ = _compute_step_acc_default(m, x)
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
    @eval begin
        function $fdm_fun(
            p::Int,
            q::Int;
            adapt::Int=1,
            condition::Real=DEFAULT_CONDITION,
            factor::Real=DEFAULT_FACTOR,
            max_range::Real=Inf,
            geom::Bool=false
        )
            _check_p_q(p, q)
            grid = $grid_fun(p)
            geom && (grid = _exponentiate_grid(grid))
            coefs, coefs_nbhd, ∇f_magnitude_mult, f_error_mult = _coefs_mults(grid, q)
            if adapt >= 1
                bound_estimator = $fdm_fun(
                    # We need to increase the order by two to be able to estimate the
                    # magnitude of the derivative of `f` in a neighbourhood of `x`.
                    p + 2,
                    p;
                    adapt=adapt - 1,
                    condition=condition,
                    factor=factor,
                    max_range=max_range,
                    geom=geom
                )
                return AdaptedFiniteDifferenceMethod{p, q, typeof(bound_estimator)}(
                    grid,
                    coefs,
                    coefs_nbhd,
                    condition,
                    factor,
                    max_range,
                    ∇f_magnitude_mult,
                    f_error_mult,
                    bound_estimator
                )
            else
                return UnadaptedFiniteDifferenceMethod{p, q}(
                    grid,
                    coefs,
                    coefs_nbhd,
                    condition,
                    factor,
                    max_range,
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
        max_range::Real=Inf,
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
- `max_range::Real=Inf`: Maximum distance that a function is evaluated from the input at
    which the derivative is estimated.
- `geom::Bool`: Use geometrically spaced points instead of linearly spaced points.

# Returns
- `FiniteDifferenceMethod`: The specified finite difference method.
        """ $fdm_fun
    end
end

_forward_grid(p::Int) = SVector{p}(0:(p - 1))

_backward_grid(p::Int) = SVector{p}((1 - p):0)

function _central_grid(p::Int)
    if isodd(p)
        return SVector{p}(div(1 - p, 2):div(p - 1, 2))
    else
        return SVector{p}((div(-p, 2):-1)..., (1:div(p, 2))...)
    end
end

function _exponentiate_grid(grid::SVector{P,Int}, base::Int=3) where P
    return sign.(grid) .* div.(base .^ abs.(grid), base)
end

function _is_symmetric(
    vec::SVector{P};
    centre_zero::Bool=false,
    negate_half::Bool=false
) where P
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
    initial_step::Real=10;
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
