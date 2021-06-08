"""
    rand_tangent([rng::AbstractRNG,] x)

Returns a arbitary tangent vector _appropriate_ for the primal value `x`.
Note that despite the name, no promises on the statistical randomness are made.
Rather it is an arbitary value, that is generated using the `rng`.
"""
rand_tangent(x) = rand_tangent(Random.GLOBAL_RNG, x)

rand_tangent(rng::AbstractRNG, x::Symbol) = NoTangent()
rand_tangent(rng::AbstractRNG, x::AbstractChar) = NoTangent()
rand_tangent(rng::AbstractRNG, x::AbstractString) = NoTangent()

rand_tangent(rng::AbstractRNG, x::Integer) = NoTangent()

# Try and make nice numbers with short decimal representations for good error messages
# while also not biasing the sample space too much
function rand_tangent(rng::AbstractRNG, x::T) where {T<:Number}
    # multiply by 9 to give a bigger range of values tested: no so tightly clustered around 0.
    return round(9 * randn(rng, T), sigdigits=5, base=2)
end
rand_tangent(rng::AbstractRNG, x::Float64) = rand(rng, -9:0.01:9)
function rand_tangent(rng::AbstractRNG, x::ComplexF64)
    return ComplexF64(rand(rng, -9:0.1:9), rand(rng, -9:0.1:9))
end

#BigFloat/MPFR is finicky about short numbers, this doesn't always work as well as it should

# multiply by 9 to give a bigger range of values tested: no so tightly clustered around 0.
rand_tangent(rng::AbstractRNG, ::BigFloat) = round(big(9 * randn(rng)), sigdigits=5, base=2)

rand_tangent(rng::AbstractRNG, x::StridedArray) = rand_tangent.(Ref(rng), x)
rand_tangent(rng::AbstractRNG, x::Adjoint) = adjoint(rand_tangent(rng, parent(x)))
rand_tangent(rng::AbstractRNG, x::Transpose) = transpose(rand_tangent(rng, parent(x)))

function rand_tangent(rng::AbstractRNG, x::T) where {T<:Tuple}
    return Tangent{T}(rand_tangent.(Ref(rng), x)...)
end

function rand_tangent(rng::AbstractRNG, xs::T) where {T<:NamedTuple}
    return Tangent{T}(; map(x -> rand_tangent(rng, x), xs)...)
end

function rand_tangent(rng::AbstractRNG, x::T) where {T}
    if !isstructtype(T)
        throw(ArgumentError("Non-struct types are not supported by this fallback."))
    end

    field_names = fieldnames(T)
    tangents = map(field_names) do field_name
        rand_tangent(rng, getfield(x, field_name))
    end
    if all(tangent isa NoTangent for tangent in tangents)
        # if none of my fields can be perturbed then I can't be perturbed
        return NoTangent()
    else
        Tangent{T}(; NamedTuple{field_names}(tangents)...)
    end
end

rand_tangent(rng::AbstractRNG, ::Type) = NoTangent()
rand_tangent(rng::AbstractRNG, ::Module) = NoTangent()
