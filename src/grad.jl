"""
    jacobian(fdm, f, x...)

Approximate the Jacobian of `f` at `x` using `fdm`. Results will be returned as a
`Matrix{<:Real}` of `size(length(y_vec), length(x_vec))` where `x_vec` is the flattened
version of `x`, and `y_vec` the flattened version of `f(x...)`. Flattening performed by
[`to_vec`](@ref).
"""
function jacobian(fdm, f, x::Vector{<:Real}; len=nothing)
    len !== nothing && Base.depwarn(
        "`len` keyword argument to `jacobian` is no longer required " *
        "and will not be permitted in the future.",
         :jacobian
    )
    ẏs = map(eachindex(x)) do n
        return fdm(zero(eltype(x))) do ε
            xn = x[n]
            x[n] = xn + ε
            ret = copy(first(to_vec(f(x))))  # copy required incase `f(x)` returns something that aliases `x`
            x[n] = xn  # Can't do `x[n] -= ϵ` as floating-point math is not associative
            return ret
        end
    end
    return (hcat(ẏs...), )
end

function jacobian(fdm, f, x; len=nothing)
    x_vec, from_vec = to_vec(x)
    return jacobian(fdm, f ∘ from_vec, x_vec; len=len)
end

function jacobian(fdm, f, xs...; len=nothing)
    return ntuple(length(xs)) do k
        jacobian(fdm, x->f(replace_arg(x, xs, k)...), xs[k]; len=len)[1]
    end
end

replace_arg(x, xs::Tuple, k::Int) = ntuple(p -> p == k ? x : xs[p], length(xs))

"""
    _jvp(fdm, f, x::Vector{<:Real}, ẋ::AbstractVector{<:Real})

Convenience function to compute `jacobian(f, x) * ẋ`.
"""
function _jvp(fdm, f, x::Vector{<:Real}, ẋ::Vector{<:Real})
    return fdm(ε -> f(x .+ ε .* ẋ), zero(eltype(x)))
end

"""
    jvp(fdm, f, xẋs::Tuple{Any, Any}...)

Compute a Jacobian-vector product with any types of arguments for which [`to_vec`](@ref)
is defined. Each 2-`Tuple` in `xẋs` contains the value `x` and its tangent `ẋ`.
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
    j′vp(fdm, f, ȳ, x...)

Compute an adjoint with any types of arguments `x` for which [`to_vec`](@ref) is defined.
"""
function j′vp(fdm, f, ȳ, x)
    x_vec, vec_to_x = to_vec(x)
    ȳ_vec, _ = to_vec(ȳ)
    return (vec_to_x(_j′vp(fdm, first ∘ to_vec ∘ f ∘ vec_to_x, ȳ_vec, x_vec)), )
end

j′vp(fdm, f, ȳ, xs...) = j′vp(fdm, xs->f(xs...), ȳ, xs)[1]

function _j′vp(fdm, f, ȳ::Vector{<:Real}, x::Vector{<:Real})
    return transpose(first(jacobian(fdm, f, x))) * ȳ
end

"""
    grad(fdm, f, xs...)

Compute the gradient of `f` for any `xs` for which [`to_vec`](@ref) is defined.
"""
grad(fdm, f, xs...) = j′vp(fdm, f, 1, xs...)  # `j′vp` with seed of 1
