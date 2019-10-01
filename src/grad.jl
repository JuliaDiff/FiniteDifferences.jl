export grad, jacobian, jvp, j′vp, to_vec
replace_arg(x, xs::Tuple, k::Int) = ntuple(p -> p == k ? x : xs[p], length(xs))

"""
    grad(fdm, f, xs...)

Approximate the gradient of `f` at `xs...` using `fdm`. Assumes that `f(xs...)` is scalar.
"""
function grad end

function grad(fdm, f, x::AbstractArray{T}) where T
    dx = similar(x)
    tmp = similar(x)
    for k in eachindex(x)
        dx[k] = fdm(zero(T)) do ϵ
            tmp .= x
            tmp[k] += ϵ
            return f(tmp)
        end
    end
    return dx
end

grad(fdm, f, x::Real) = fdm(f, x)
grad(fdm, f, x::Tuple) = grad(fdm, (xs...)->f(xs), x...)

function grad(fdm, f, d::Dict{K, V}) where {K, V}
    dd = Dict{K, V}()
    for (k, v) in d
        function f′(x)
            tmp = copy(d)
            tmp[k] = x
            return f(tmp)
        end
        dd[k] = grad(fdm, f′, v)
    end
    return dd
end

function grad(fdm, f, xs...)
    return ntuple(length(xs)) do k
        grad(fdm, x->f(replace_arg(x, xs, k)...), xs[k])
    end
end

"""
    jacobian(fdm, f, xs::Union{Real, AbstractArray{<:Real}}; dim::Int=length(f(x)))

Approximate the Jacobian of `f` at `x` using `fdm`. `f(x)` must be a length `D` vector. If
`D` is not provided, then `f(x)` is computed once to determine the output size.
"""
function jacobian(fdm, f, x::Union{T, AbstractArray{T}}; dim::Int=length(f(x))) where {T <: Real}
    J = Matrix{float(T)}(undef, dim, length(x))
    for d in 1:dim
        gs = grad(fdm, x->f(x)[d], x)
        for k in 1:length(x)
            J[d, k] = gs[k]
        end
    end
    return J
end

function jacobian(fdm, f, xs...; dim::Int=length(f(xs...)))
    return ntuple(length(xs)) do k
        jacobian(fdm, x->f(replace_arg(x, xs, k)...), xs[k]; dim=dim)
    end
end

"""
    _jvp(fdm, f, x::Vector{<:Number}, ẋ::AbstractVector{<:Number})

Convenience function to compute `jacobian(f, x) * ẋ`.
"""
_jvp(fdm, f, x::Vector{<:Number}, ẋ::AV{<:Number}) = fdm(ε -> f(x .+ ε .* ẋ), zero(eltype(x)))

"""
    _j′vp(fdm, f, ȳ::AbstractVector{<:Number}, x::Vector{<:Number})

Convenience function to compute `transpose(jacobian(f, x)) * ȳ`.
"""
_j′vp(fdm, f, ȳ::AV{<:Number}, x::Vector{<:Number}) = transpose(jacobian(fdm, f, x; dim=length(ȳ))) * ȳ

"""
    jvp(fdm, f, x, ẋ)

Compute a Jacobian-vector product with any types of arguments for which [`to_vec`](@ref)
is defined.
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

Compute an adjoint with any types of arguments for which [`to_vec`](@ref) is defined.
"""
function j′vp(fdm, f, ȳ, x)
    x_vec, vec_to_x = to_vec(x)
    ȳ_vec, _ = to_vec(ȳ)
    return vec_to_x(_j′vp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], ȳ_vec, x_vec))
end
j′vp(fdm, f, ȳ, xs...) = j′vp(fdm, xs->f(xs...), ȳ, xs)

"""
    to_vec(x) -> Tuple{<:AbstractVector, <:Function}

Transform `x` into a `Vector`, and return a closure which inverts the transformation.
"""
to_vec(x::Number) = ([x], first)

# Vectors
to_vec(x::Vector{<:Number}) = (x, identity)
function to_vec(x::Vector)
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    return vcat(x_vecs...), function(x_vec)
        sz = cumsum([map(length, x_vecs)...])
        return [backs[n](x_vec[sz[n]-length(x_vecs[n])+1:sz[n]]) for n in eachindex(x)]
    end
end

# Arrays
to_vec(x::Array{<:Number}) = vec(x), x_vec->reshape(x_vec, size(x))
function to_vec(x::Array)
    x_vec, back = to_vec(reshape(x, :))
    return x_vec, x_vec->reshape(back(x_vec), size(x))
end

# AbstractArrays
function to_vec(x::T) where {T<:LinearAlgebra.AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    return x_vec, x_vec->T(reshape(back(x_vec), size(x)))
end
to_vec(x::Symmetric) = vec(Matrix(x)), x_vec->Symmetric(reshape(x_vec, size(x)))
to_vec(X::Diagonal) = vec(Matrix(X)), x_vec->Diagonal(reshape(x_vec, size(X)...))

function to_vec(X::Transpose)
    return vec(Matrix(X)), x_vec->Transpose(permutedims(reshape(x_vec, size(X))))
end
function to_vec(X::Adjoint)
    return vec(Matrix(X)), x_vec->Adjoint(conj!(permutedims(reshape(x_vec, size(X)))))
end

# Non-array data structures

function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum([map(length, x_vecs)...])
    return vcat(x_vecs...), function(v)
        return ntuple(n->x_backs[n](v[sz[n]-length(x_vecs[n])+1:sz[n]]), length(x))
    end
end

# Convert to a vector-of-vectors to make use of existing functionality.
function to_vec(d::Dict)
    d_vec_vec = [val for val in values(d)]
    d_vec, back = to_vec(d_vec_vec)
    return d_vec, function(v)
        v_vec_vec = back(v)
        return Dict([(key, v_vec_vec[n]) for (n, key) in enumerate(keys(d))])
    end
end
