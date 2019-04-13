"""
    grad(fdm, f, x::AbstractVector)

Approximate the gradient of `f` at `x` using `fdm`. Assumes that `f(x)` is scalar.
"""
function grad(fdm, f, x::Vector{T}) where T<:Real
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
    jacobian(fdm, f, x::AbstractVector{<:Real}, D::Int)
    jacobian(fdm, f, x::AbstractVector{<:Real})

Approximate the Jacobian of `f` at `x` using `fdm`. `f(x)` must be a length `D` vector. If
`D` is not provided, then `f(x)` is computed once to determine the output size.
"""
function jacobian(fdm, f, x::Vector{T}, D::Int) where {T<:Real}
    J = Matrix{T}(undef, D, length(x))
    for d in 1:D
        J[d, :] = grad(fdm, x->f(x)[d], x)
    end
    return J
end
jacobian(fdm, f, x::Vector{<:Real}) = jacobian(fdm, f, x, length(f(x)))

"""
    _jvp(fdm, f, x::Vector{<:Real}, ẋ::AbstractVector{<:Real})

Convenience function to compute `jacobian(f, x) * ẋ`.
"""
_jvp(fdm, f, x::Vector{<:Real}, ẋ::AV{<:Real}) = jacobian(fdm, f, x) * ẋ

"""
    _j′vp(fdm, f, ȳ::AbstractVector{<:Real}, x::Vector{<:Real})

Convenience function to compute `jacobian(f, x)' * ȳ`.
"""
_j′vp(fdm, f, ȳ::AV{<:Real}, x::Vector{<:Real}) = jacobian(fdm, f, x, length(ȳ))' * ȳ

"""
    jvp(fdm, f, x, ẋ)

Compute a Jacobian-vector product with any types of arguments for which `to_vec` is defined.
"""
function jvp(fdm, f, (x, ẋ)::Tuple{Any, Any})
    x_vec, vec_to_x = to_vec(x)
    _, vec_to_y = to_vec(f(x))
    return vec_to_y(_jvp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], x_vec, to_vec(ẋ)[1]))
end
function jvp(fdm, f, xẋs::Tuple{Any, Any}...)
    x, ẋ = collect(zip(xẋs...))
    return jvp(fdm, xs->f(xs...), (x, ẋ))
end

"""
    j′vp(fdm, f, ȳ, x...)

Compute an adjoint with any types of arguments for which `to_vec` is defined.
"""
function j′vp(fdm, f, ȳ, x)
    x_vec, vec_to_x = to_vec(x)
    ȳ_vec, _ = to_vec(ȳ)
    return vec_to_x(_j′vp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], ȳ_vec, x_vec))
end
j′vp(fdm, f, ȳ, xs...) = j′vp(fdm, xs->f(xs...), ȳ, xs)

# Transform `x` into a vector, and return a closure which inverts the transformation.
to_vec(x::Real) = ([x], x->x[1])

# Arrays.
to_vec(x::Vector{<:Real}) = (x, identity)
to_vec(x::Array) = vec(x), x_vec->reshape(x_vec, size(x))

# AbstractArrays.
function to_vec(x::T) where {T<:LinearAlgebra.AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    return x_vec, x_vec->T(reshape(back(x_vec), size(x)))
end
to_vec(x::Symmetric) = vec(Matrix(x)), x_vec->Symmetric(reshape(x_vec, size(x)))
to_vec(X::Diagonal) = vec(Matrix(X)), x_vec->Diagonal(reshape(x_vec, size(X)...))

# Non-array data structures.
function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum([map(length, x_vecs)...])
    return vcat(x_vecs...), function(v)
        return ([x_backs[n](v[sz[n]-length(x[n])+1:sz[n]]) for n in 1:length(x)]...,)
    end
end
