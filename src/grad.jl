export grad, jacobian, jvp, j′vp, to_vec
replace_arg(x, xs::Tuple, k::Int) = ntuple(p -> p == k ? x : xs[p], length(xs))

"""
    _jvp(fdm, f, x::Vector{<:Number}, ẋ::AbstractVector{<:Number})

Convenience function to compute `jacobian(f, x) * ẋ`.
"""
_jvp(fdm, f, x::Vector{<:Number}, ẋ::AV{<:Number}) = fdm(ε -> f(x .+ ε .* ẋ), zero(eltype(x)))

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
    return jvp(fdm, xs->f(xs...)[1], (x, ẋ))
end

"""
    jacobian(fdm, f, xs::Union{Real, AbstractArray{<:Real}})

Approximate the Jacobian of `f` at `x` using `fdm`.
"""
function jacobian(fdm, f, x::Union{T, AbstractArray{T}}) where {T <: Number}

    # Allocate for the perturbation.
    ẋ = zero(x)

    # Compute the first element so that we know what the output type is.
    ẋ[1] = one(float(T))
    ẏ = jvp(fdm, f, (x, ẋ))
    ẋ[1] = zero(float(T))

    # Allocate for the sensitivities.
    @show typeof(ẏ)
    @assert ẏ isa Array{<:Number}
    ẏs = Vector{typeof(ẏ)}(undef, length(x))
    ẏs[1] = ẏ

    # Iterate over the remainder of the input elements.
    for n in 2:length(x)
        ẋ[n] = one(float(T))
        ẏs[n] = jvp(fdm, f, (x, ẋ))
        ẋ[n] = zero(float(T))
    end

    return (hcat(ẏs...), )
end

function jacobian(fdm, f, xs...)
    return ntuple(length(xs)) do k
        jacobian(fdm, x->f(replace_arg(x, xs, k)...), xs[k])[1]
    end
end



"""
    grad(fdm, f, xs...)

Approximate the gradient of `f` at `xs...` using `fdm`. Assumes that `f(xs...)` is scalar.
"""
function grad end

function _grad(fdm, f, x::AbstractArray{T}) where T <: Number
    # x must be mutable, we will mutate it and then mutate it back.
    dx = similar(x)
    for k in eachindex(x)
        dx[k] = fdm(zero(T)) do ϵ
            xk = x[k]
            x[k] = xk +  ϵ
            ret = f(x)
            x[k] = xk  # Can't do `x[k] -= ϵ` as floating-point math is not associative
            return ret
        end
    end
    return (dx, )
end

grad(fdm, f, x::Array{<:Number}) = _grad(fdm, f, x)
# Fallback for when we don't know `x` will be mutable:
grad(fdm, f, x::AbstractArray{<:Number}) = _grad(fdm, f, similar(x).=x)

grad(fdm, f, x::Real) = (fdm(f, x), )
grad(fdm, f, x::Tuple) = (grad(fdm, (xs...)->f(xs), x...), )

function grad(fdm, f, d::Dict{K, V}) where {K, V}
    ∇d = Dict{K, V}()
    for (k, v) in d
        dk = d[k]
        function f′(x)
            d[k] = x
            return f(d)
        end
        ∇d[k] = grad(fdm, f′, v)[1]
        d[k] = dk
    end
    return (∇d, )
end

function grad(fdm, f, x)
    v, back = to_vec(x)
    return (back(grad(fdm, x->f(back(v)), v)), )
end

function grad(fdm, f, xs...)
    return ntuple(length(xs)) do k
        grad(fdm, x->f(replace_arg(x, xs, k)...), xs[k])[1]
    end
end


"""
    _j′vp(fdm, f, ȳ::AbstractVector{<:Number}, x::Vector{<:Number})

Convenience function to compute `transpose(jacobian(f, x)) * ȳ`.
"""
_j′vp(fdm, f, ȳ::AV{<:Number}, x::Vector{<:Number}) = transpose(jacobian(fdm, f, x)[1]) * ȳ


"""
    j′vp(fdm, f, ȳ, x...)

Compute an adjoint with any types of arguments for which [`to_vec`](@ref) is defined.
"""
function j′vp(fdm, f, ȳ, x)
    x_vec, vec_to_x = to_vec(x)
    ȳ_vec, _ = to_vec(ȳ)
    return (vec_to_x(_j′vp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], ȳ_vec, x_vec)), )
end
j′vp(fdm, f, ȳ, xs...) = j′vp(fdm, xs->f(xs...), ȳ, xs)[1]

"""
    to_vec(x) -> Tuple{<:AbstractVector, <:Function}

Transform `x` into a `Vector`, and return a closure which inverts the transformation.
"""
function to_vec(x::Number)
    function Number_from_vec(x_vec)
        return first(x_vec)
    end
    return [x], Number_from_vec
end

# Vectors
to_vec(x::Vector{<:Number}) = (x, identity)
function to_vec(x::Vector)
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = cumsum([map(length, x_vecs)...])
        return [backs[n](x_vec[sz[n]-length(x_vecs[n])+1:sz[n]]) for n in eachindex(x)]
    end
    return vcat(x_vecs...), Vector_from_vec
end

# Arrays
function to_vec(x::Array{<:Number})
    function Array_from_vec(x_vec)
        return reshape(x_vec, size(x))
    end
    return vec(x), Array_from_vec
end

function to_vec(x::Array)
    x_vec, back = to_vec(reshape(x, :))
    function Array_from_vec(x_vec)
        return reshape(back(x_vec), size(x))
    end
    return x_vec, Array_from_vec
end

# AbstractArrays
function to_vec(x::T) where {T<:LinearAlgebra.AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    function AbstractTriangular_from_vec(x_vec)
        return T(reshape(back(x_vec), size(x)))
    end
    return x_vec, AbstractTriangular_from_vec
end

function to_vec(x::Symmetric)
    function Symmetric_from_vec(x_vec)
        return Symmetric(reshape(x_vec, size(x)))
    end
    return vec(Matrix(x)), Symmetric_from_vec
end

function to_vec(X::Diagonal)
    function Diagonal_from_vec(x_vec)
        return Diagonal(reshape(x_vec, size(X)...))
    end
    return vec(Matrix(X)), Diagonal_from_vec
end

function to_vec(X::Transpose)
    function Transpose_from_vec(x_vec)
        return Transpose(permutedims(reshape(x_vec, size(X))))
    end
    return vec(Matrix(X)), Transpose_from_vec
end

function to_vec(X::Adjoint)
    function Adjoint_from_vec(x_vec)
        return Adjoint(conj!(permutedims(reshape(x_vec, size(X)))))
    end
    return vec(Matrix(X)), Adjoint_from_vec
end

# Non-array data structures

function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum([map(length, x_vecs)...])
    function Tuple_from_vec(v)
        return ntuple(n->x_backs[n](v[sz[n]-length(x_vecs[n])+1:sz[n]]), length(x))
    end
    return vcat(x_vecs...), Tuple_from_vec
end

# Convert to a vector-of-vectors to make use of existing functionality.
function to_vec(d::Dict)
    d_vec_vec = [val for val in values(d)]
    d_vec, back = to_vec(d_vec_vec)
    function Dict_from_vec(v)
        v_vec_vec = back(v)
        return Dict([(key, v_vec_vec[n]) for (n, key) in enumerate(keys(d))])
    end
    return d_vec, Dict_from_vec
end
