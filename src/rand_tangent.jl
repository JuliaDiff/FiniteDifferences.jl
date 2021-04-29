"""
    rand_tangent([rng::AbstractRNG,] x)

Returns a randomly generated tangent vector appropriate for the primal value `x`.
"""
rand_tangent(x) = rand_tangent(Random.GLOBAL_RNG, x)

rand_tangent(rng::AbstractRNG, x::Symbol) = DoesNotExist()
rand_tangent(rng::AbstractRNG, x::AbstractChar) = DoesNotExist()
rand_tangent(rng::AbstractRNG, x::AbstractString) = DoesNotExist()

rand_tangent(rng::AbstractRNG, x::Integer) = DoesNotExist()

rand_tangent(rng::AbstractRNG, x::T) where {T<:Number} = randn(rng, T)

rand_tangent(rng::AbstractRNG, x::StridedArray) = rand_tangent.(Ref(rng), x)

function rand_tangent(rng::AbstractRNG, x::T) where {T<:Tuple}
    return Composite{T}(rand_tangent.(Ref(rng), x)...)
end

function rand_tangent(rng::AbstractRNG, xs::T) where {T<:NamedTuple}
    return Composite{T}(; map(x -> rand_tangent(rng, x), xs)...)
end

function rand_tangent(rng::AbstractRNG, x::T) where {T}
    if !isstructtype(T)
        throw(ArgumentError("Non-struct types are not supported by this fallback."))
    end

    field_names = fieldnames(T)
    if length(field_names) > 0
        tangents = map(field_names) do field_name
            rand_tangent(rng, getfield(x, field_name))
        end
        if all(tangent isa DoesNotExist for tangent in tangents)
            # if none of my fields can be perturbed then I can't be perturbed
            return DoesNotExist()
        else
            Composite{T}(; NamedTuple{field_names}(tangents)...)
        end
    else
        return NO_FIELDS
    end
end
