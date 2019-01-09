"""
    grad(fdm, f, x::AbstractVector)

Approximate the gradient of `f` at `x` using `fdm`. Assumes that `f(x)` is scalar.
"""
function grad(fdm, f, x::AbstractArray{T}) where T<:Real
    v, dx, tmp = fill(zero(T), size(x)), similar(x), similar(x)
    for n in eachindex(x)
        v[n] = one(T)
        dx[n] = fdm(function(ϵ)
                tmp .= x .+ ϵ .* v
                return f(tmp)
            end,
            zero(T),
        )
        v[n] = zero(T)
    end
    return dx
end

"""
    jacobian(fdm, f, x::AbstractVector, D::Int)
    jacobian(fdm, f, x::AbstractVector)

Approximate the Jacobian of `f` at `x` using `fdm`. `f(x)` must be a length `D` vector. If
`D` is not provided, then `f(x)` is computed once to determine the output size.
"""
function jacobian(fdm, f, x::AbstractVector{T}, D::Int) where {T<:Real}
    J = Matrix{T}(undef, D, length(x))
    for d in 1:D
        J[d, :] = grad(fdm, x->f(x)[d], x)
    end
    return J
end
jacobian(fdm, f, x::AbstractVector{<:Real}) = jacobian(fdm, f, x, length(f(x)))

"""
    adjoint(fdm, f, ȳ::AbstractVector, x::AbstractVector)

Convenience function to compute `ȳ' * jacobian(f, x)`.
"""
function Base.adjoint(fdm, f, ȳ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return jacobian(fdm, f, x, length(ȳ))' * ȳ
end
