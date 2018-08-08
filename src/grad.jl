"""
    grad(fdm, f, x::AbstractVector)

Approximate the gradient of `f` at `x` using `fdm`. Assumes that `f(x)` is scalar.
"""
function grad(fdm, f, x::AbstractArray{T}) where T
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
