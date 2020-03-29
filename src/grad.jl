export grad, jacobian, jvp, j′vp, to_vec

replace_arg(x, xs::Tuple, k::Int) = ntuple(p -> p == k ? x : xs[p], length(xs))

"""
    jacobian(fdm, f, x...)

Approximate the Jacobian of `f` at `x` using `fdm`. Results will be returned as a
`Matrix{<:Number}` of `size(length(y_vec), length(x_vec))` where `x_vec` is the flattened
version of `x`, and `y_vec` the flattened version of `f(x...)`. Flattening performed by
`to_vec`.
"""
function jacobian(fdm, f, x::Vector{<:Number})

    # Construct a transformation of f that outputs Vector{<:Number}.
    f_vec = first ∘ to_vec ∘ f

    # Compute the first element so that we know what the output type is.
    ẏ = fdm(zero(eltype(x))) do ε
        x1 = x[1]
        x[1] = x1 + ε
        ret = f_vec(x)
        x[1] = x1
        return ret
    end

    # Allocate for the sensitivities.
    @assert ẏ isa Vector{<:Number}
    ẏs = Vector{typeof(ẏ)}(undef, length(x))
    ẏs[1] = ẏ

    # Iterate over the remainder of the input elements.
    for n in 2:length(x)
        ẏs[n] = fdm(zero(eltype(x))) do ε
            xn = x[n]
            x[n] = xn + ε
            ret = f_vec(x)
            x[n] = xn
            return ret
        end
    end

    # Spit out a 1-Tuple containing a Matrix{<:Number}.
    return (hcat(ẏs...), )
end

function jacobian(fdm, f, x)
    x_vec, from_vec = to_vec(x)
    return jacobian(fdm, f ∘ from_vec, x_vec)
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
            x[k] = xk + ϵ
            ret = f(x)
            x[k] = xk  # Can't do `x[k] -= ϵ` as floating-point math is not associative
            return ret
        end
    end
    return (dx, )
end

# function jacobian(fdm, f, x)
#     x_vec, from_vec = to_vec(x)
#     return jacobian(fdm, f ∘ from_vec, x_vec)
# end

# function jacobian(fdm, f, xs...)
#     return ntuple(length(xs)) do k
#         jacobian(fdm, x->f(replace_arg(x, xs, k)...), xs[k])[1]
#     end
# end

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
    _jvp(fdm, f, x::Vector{<:Number}, ẋ::AbstractVector{<:Number})

Convenience function to compute `jacobian(f, x) * ẋ`.
"""
function _jvp(fdm, f, x::Vector{<:Number}, ẋ::Vector{<:Number})
    return fdm(ε -> f(x .+ ε .* ẋ), zero(eltype(x)))
end

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
    j′vp(fdm, f, ȳ::AbstractVector{<:Number}, x::Vector{<:Number})

Convenience function to compute `transpose(jacobian(f, x)) * ȳ`.
"""
function _j′vp(fdm, f, ȳ::Vector{<:Number}, x::Vector{<:Number})
    return transpose(jacobian(fdm, f, x)[1]) * ȳ
end

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
