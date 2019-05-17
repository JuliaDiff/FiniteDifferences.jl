export FDMReport, fdm, backward_fdm, forward_fdm, central_fdm

"""
    FDM.DEFAULT_CONDITION

The default [condition number](https://en.wikipedia.org/wiki/Condition_number) used when
computing bounds. It provides amplification of the ∞-norm when passed to the function's
derivatives.
"""
const DEFAULT_CONDITION = 100

"""
    FDM.TINY

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

abstract type FDMethod end

for D in (:Forward, :Backward, :Central)
    gridf = Symbol(lowercase(String(D)), "_grid")
    @eval begin
        struct $D{G<:AbstractVector, C<:AbstractVector} <: FDMethod
            p::Int
            q::Int
            grid::G
            coefs::C
        end
        function $D(p::Integer, q::Integer)
            q < p || throw(ArgumentError("order of the method must be strictly greater " *
                                         "than that of the derivative"))
            # Check whether the method can be computed. We require the factorial of the
            # method order to be computable with regular `Int`s, but `factorial` will overflow
            # after 20, so 20 is the largest we can allow.
            p > 20 && throw(ArgumentError("order of the method is too large to be computed"))

            grid = $gridf(p)

            C = [g^i for i in 0:(p - 1), g in grid]
            x = zeros(Int, p)
            x[q + 1] = factorial(q)
            coefs = C \ x

            return $D{typeof(grid), typeof(coefs)}(p, q, grid, coefs)
        end
        (d::$D)(f, x; kwargs...) = fdm(d, f, x; kwargs...)
    end
end

estimate_bound(x, cond) = cond * maximum(abs, x) + TINY

function fdm(
    m::M,
    f,
    x;
    condition=DEFAULT_CONDITION,
    bound=estimate_bound(f(x), condition),
    eps=Base.eps(bound),
    adapt=1,
    report=false,
) where M<:FDMethod
    eps > 0 || throw(ArgumentError("eps must be positive, got $eps"))
    bound > 0 || throw(ArgumentError("bound must be positive, got $bound"))
    0 <= adapt < 20 - m.p || throw(ArgumentError("can't perform $adapt adaptation steps"))

    p = m.p
    q = m.q
    grid = m.grid
    coefs = m.coefs

    # Adaptively compute the bound on the function and derivative values, if applicable.
    if 0 < adapt < p
        newm = (M.name.wrapper)(p + 1, p)
        dfdx, = fdm(newm, f, x; eps=eps, bound=bound, adapt=(adapt - 1))
        bound = estimate_bound(dfdx, condition)
    end

    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = eps * sum(abs, coefs)
    C₂ = bound * sum(n->abs(coefs[n] * grid[n]^p), eachindex(coefs)) / factorial(p)
    ĥ = (q / (p - q) * C₁ / C₂) ^ (1 / p)

    # Estimate the accuracy of the method.
    accuracy = ĥ^(-q) * C₁ + ĥ^(p - q) * C₂

    # Estimate the value of the derivative.
    dfdx = sum(i->coefs[i] * f(x + ĥ * grid[i]), eachindex(grid)) / ĥ^q

    if report
        return dfdx, FDMReport(p, q, grid, coefs, eps, bound, ĥ, accuracy)
    else
        return dfdx
    end
end

"""
    FDMReport

Details of a finite difference method to estimate a derivative. Instances of `FDMReport`
`Base.show` nicely.

# Fields
- `p::Int`: Order of the method.
- `q::Int`: Order of the derivative that is estimated.
- `grid::Vector{<:Real}`: Relative spacing of samples of `f` that are used by the method.
- `coefs::Vector{<:Real}`: Weights of the samples of `f`.
- `ε::Real`: Absolute roundoff error of the function evaluations.
- `M::Real`: Assumed upper bound of `f` and all its derivatives at `x`.
- `ĥ::Real`: Step size.
- `err::Real`: Estimated absolute accuracy.
"""
struct FDMReport{Tε, TM, Tĥ, Tacc}
    p::Int
    q::Int
    grid::Vector{<:Real}
    coefs::Vector{<:Real}
    ε::Tε
    M::TM
    ĥ::Tĥ
    acc::Tacc
end
function Base.show(io::IO, x::FDMReport)
    @printf io "FDMReport:\n"
    @printf io "  order of method:       %d\n" x.p
    @printf io "  order of derivative:   %d\n" x.q
    @printf io "  grid:                  %s\n" x.grid
    @printf io "  coefficients:          %s\n" x.coefs
    @printf io "  roundoff error:        %.2e\n" x.ε
    @printf io "  bounds on derivatives: %.2e\n" x.M
    @printf io "  step size:             %.2e\n" x.ĥ
    @printf io "  accuracy:              %.2e\n" x.acc
end

"""
    function fdm(
        grid::Vector{<:Real},
        q::Int;
        ε::Real=eps(),
        M::Real=1,
        report::Bool=false
    )

Construct a function `method(f, x::Real, h::Real=ĥ)` that takes in a
function `f`, a point `x` in the domain of `f`, and optionally a step size `h`, and
estimates the `q`'th order derivative of `f` at `x` with a `length(grid)`'th order
finite difference method.

# Arguments
- `grid::Vector{<:Real}`: Relative spacing of samples of `f` that are used by the method.
    The length of `grid` determines the order of the method.
- `q::Int`: Order of the derivative to estimate. `q` must be strictly less than the order
    of the method.

# Keywords
- `ε::Real=eps()`: Absolute roundoff error on the function evaluations.
- `M::Real=1`: Upper bound on `f` and all its derivatives.
- `::Bool=false`: Also return an instance of `FDMReport` containing information
    about the method constructed.
"""
function fdm(grid::AbstractVector{T}, q::Int, ::Val{false}; ε::Real=eps(), M::Real=one(T),
    ) where {T<:Real}
    method, p, q, grid, coefs, ε, M, ĥ, acc = _fdm(grid, q, ε, M)
    return method
end
function fdm(grid::AbstractVector{T}, q::Int, ::Val{true}; ε::Real=eps(), M::Real=one(T),
    ) where {T<:Real}
    method, p, q, grid, coefs, ε, M, ĥ, acc = _fdm(grid, q, ε, M)
    return method, FDMReport(p, q, grid, coefs, ε, M, ĥ, acc)
end
function fdm(grid::AbstractVector{T}, q::Int; ε::Real=eps(), M::Real=one(T)) where T<:Real
    return fdm(grid, q, Val(false); ε=ε, M=M)
end

function _fdm(grid::AbstractVector{T}, q::Int, ε::Real, M::Real) where T
    p = length(grid)  # Order of the method.
    q < p || throw(ArgumentError("Order of the method must be strictly greater than that " *
                                 "of the derivative."))

    # Check whether the method can be computed. We require the factorial of the method
    # order to be computable with regular `Int`s, but `factorial` will overflow after 20,
    # so 20 is the largest we can allow.
    p > 20 && throw(ArgumentError("Order of the method is too large to be computed"))
    factp = factorial(p)

    # Compute the coefficients of the FDM.
    C = [grid[n]^i for i in 0:p - 1, n in eachindex(grid)]
    x = zeros(Int, p)
    x[q+1] = factorial(q)
    coefs = C \ x

    # Set the step size by minimising an upper bound on the error of the estimate.
    C₁ = ε * sum(abs, coefs)
    C₂ = M * sum(n->abs(coefs[n] * grid[n]^p), eachindex(coefs)) / factp
    ĥ = (q / (p - q) * C₁ / C₂) ^ (1 / p)

    # Estimate the accuracy of the method.
    acc = ĥ^(-q) * C₁ + ĥ^(p - q) * C₂

    # Construct the FDM.
    function method(f, x::Real=zero(T), h::Real=ĥ)
        return sum(n->coefs[n] * f(x + h * grid[n]), eachindex(grid)) / h^q
    end
    return method, p, q, grid, coefs, ε, M, ĥ, acc
end


"""
    backward_fdm(p::Int, ...)

Construct a backward finite difference method of order `p`. See `fdm` for further details.

# Arguments
- `p::Int`: Order of the method.

Further takes, in the following order, the arguments `q`, `ε`, `M`, and `report` from `fdm`.
"""
backward_fdm(p::Int, args...; kws...) = fdm(1 - p:0, args...; kws...)

"""
    forward_fdm(p::Int, ...)

Construct a forward finite difference method of order `p`. See `fdm` for further details.

# Arguments
- `p::Int`: Order of the method.

Further takes, in the following order, the arguments `q`, `ε`, `M`, and `report` from `fdm`.
"""
forward_fdm(p::Int, args...; kws...) = fdm(0:p - 1, args...; kws...)

"""
    central_fdm(p::Int, ...)

Construct a central finite difference method of order `p`. See `fdm` for further details.

# Arguments
- `p::Int`: Order of the method.

Further takes, in the following order, the arguments `q`, `ε`, `M`, and `report` from `fdm`.
"""
function central_fdm(p::Int, args...; kws...)
    return isodd(p) ? fdm(Int(-(p - 1) / 2):Int((p - 1) / 2), args...; kws...) :
                      fdm([Int(-p/2):-1; 1:Int(p/2)], args...; kws...)
end
