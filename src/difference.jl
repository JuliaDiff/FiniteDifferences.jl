"""
    difference(ε::Real, y::T, x::T) where {T}

Computes `dx` where `dx` is defined s.t.
```julia
y = x + ε * dx
```
where `dx` is a valid tangent type for `x`.

If `(y - x) / ε` is defined, then this operation is equivalent to doing that. For functions
where these operations aren't defined, `difference` can still be defined without commiting
type piracy while `-` and `/` cannot.
"""
difference(::Real, ::T, ::T) where {T<:Symbol} = NoTangent()
difference(::Real, ::T, ::T) where {T<:AbstractChar} = NoTangent()
difference(::Real, ::T, ::T) where {T<:AbstractString} = NoTangent()
difference(::Real, ::T, ::T) where {T<:Integer} = NoTangent()

difference(ε::Real, y::T, x::T) where {T<:Number} = (y - x) / ε

# we are a bit more relaced for AbstractArrays as they naturally represent a vector space
difference(ε::Real, y::AbstractArray, x) = difference.(ε, y, x)
# resolve ambiguity
difference(ε::Real, y::T, x::T) where {T<:AbstractArray} = difference.(ε, y, x)

function difference(ε::Real, y::T, x::T) where {T<:Tuple}
    return Tangent{T}(difference.(ε, y, x)...)
end

function difference(ε::Real, ys::T, xs::T) where {T<:NamedTuple}
    return Tangent{T}(; map((y, x) -> difference(ε, y, x), ys, xs)...)
end

function difference(ε::Real, y::T, x::T) where {T}
    if !isstructtype(T)
        throw(ArgumentError("Non-struct types are not supported by this fallback."))
    end

    field_names = fieldnames(T)
    if length(field_names) > 0
        tangents = map(field_names) do field_name
            difference(ε, getfield(y, field_name), getfield(x, field_name))
        end
        return Tangent{T}(; NamedTuple{field_names}(tangents)...)
    else
        return NO_FIELDS
    end
end
