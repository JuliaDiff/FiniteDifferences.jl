"""
    to_vec(x)

Transform `x` into a `Vector`, and return the vector, and a closure which inverts the
transformation.
"""
function to_vec(x::Real)
    function Real_from_vec(x_vec)
        return first(x_vec)
    end
    return [x], Real_from_vec
end

function to_vec(z::Complex)
    function Complex_from_vec(z_vec)
        return Complex(z_vec[1], z_vec[2])
    end
    return [real(z), imag(z)], Complex_from_vec
end

# Base case -- if x is already a Vector{<:Real} there's no conversion necessary.
to_vec(x::Vector{<:Real}) = (x, identity)

function non_differentiable_type_to_vec(x)
    non_differentiable_from_vec(v) = x
    return Bool[], non_differentiable_from_vec
end

to_vec(x::Char) = non_differentiable_type_to_vec(x)
to_vec(x::String) = non_differentiable_type_to_vec(x)
to_vec(x::Integer) = non_differentiable_type_to_vec(x)
to_vec(x::Complex{<:Integer}) = non_differentiable_type_to_vec(x)
to_vec(x::AbstractVector{<:Integer}) = non_differentiable_type_to_vec(x)

# get around the constructors and make the type directly
# Note this is moderately evil accessing julia's internals
if VERSION >= v"1.3"
    @generated function _force_construct(T, args...)
        return Expr(:splatnew, :T, :args)
    end
else
    @generated function _force_construct(T, args...)
        return Expr(:new, :T, Any[:(args[$i]) for i in 1:length(args)]...)
    end
end

# Default method for any composite type. This is always going to be correct for
# composite types with the default inner constructor, but may require tweaking if the
# inner constructor places additional restrictions on the values that the arguments can be.
function to_vec(x::T) where {T}
    Base.isstructtype(T) || throw(error("Expected a struct type"))

    # Singleton types - segfaults without this.
    isempty(fieldnames(T)) && return (Bool[], _ -> x)

    val_vecs_and_backs = [to_vec(getfield(x, name)) for name in fieldnames(T)]
    vals = first.(val_vecs_and_backs)
    backs = last.(val_vecs_and_backs)

    v, vals_from_vec = to_vec(vals)
    function structtype_from_vec(v::Vector{<:Real})
        val_vecs = vals_from_vec(v)
        values = map((b, v) -> b(v), backs, val_vecs)
        if T <: Tuple
            return (values..., )
        else
            try
                T(values...)
            catch MethodError
                return _force_construct(T, values...)
            end
        end
    end
    return v, structtype_from_vec
end

# Technically `isstructtype`, but has annoying internal fields. Would ideally not require
# this method, but because a `Dict` isn't a plain container type, it
function to_vec(d::Dict)
    d_vec, back = to_vec(collect(values(d)))
    function Dict_from_vec(v)
        v_vec_vec = back(v)
        return Dict(key => v_vec_vec[n] for (n, key) in enumerate(keys(d)))
    end
    return d_vec, Dict_from_vec
end

function to_vec(x::DenseVector)
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = cumsum(map(length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
        return oftype(x, x_Vec)
    end
    # handle empty x
    x_vec = isempty(x_vecs) ? eltype(eltype(x_vecs))[] : reduce(vcat, x_vecs)
    return x_vec, Vector_from_vec
end

function to_vec(x::DenseArray)
    x_vec, from_vec = to_vec(vec(x))

    function Array_from_vec(x_vec)
        return oftype(x, reshape(from_vec(x_vec), size(x)))
    end

    return x_vec, Array_from_vec
end

# Factorizations

function to_vec(x::S) where {U, S <: Union{LinearAlgebra.QRCompactWYQ{U}, LinearAlgebra.QRCompactWY{U}}}
    # x.T is composed of upper triangular blocks. The subdiagonals elements
    # of the blocks are abitrary. We make sure to set all of them to zero
    # to avoid NaN.
    blocksize, cols = size(x.T)
    T = zeros(U, blocksize, cols)

    for i in 0:div(cols - 1, blocksize)
        used_cols = i * blocksize
        n = min(blocksize, cols - used_cols)
        T[1:n, (1:n) .+ used_cols] = UpperTriangular(view(x.T, 1:n, (1:n) .+ used_cols))
    end

    x_vec, back = to_vec([x.factors, T])

    function QRCompact_from_vec(v)
        factors, Tback = back(v)
        return S(factors, Tback)
    end

    return x_vec, QRCompact_from_vec
end


"""
    to_vec_tangent(x)

Transform `x` into a `Vector`, and return the vector, and a closure which builds a
`Tangent`.
"""
function to_vec_tangent(x::Real)
    Real_tangent_from_vec(x_vec) = first(x_vec)
    return [x], Real_tangent_from_vec
end

function to_vec_tangent(z::Complex)
    Complex_tangent_from_vec(z_vec) = Complex(z_vec[1], z_vec[2])
    return [real(z), imag(z)], Complex_tangent_from_vec
end

# Base case -- if x is already a Vector{<:Real} there's no conversion necessary.
to_vec_tangent(x::Vector{<:Real}) = (x, identity)

function to_vec_tangent(x::Vector)
    x_vecs_and_backs = map(to_vec_tangent, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_Tangent_from_vec(x_vec)
        sz = cumsum(map(length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
        return x_Vec
    end

    # handle empty x
    x_vec = isempty(x_vecs) ? eltype(eltype(x_vecs))[] : reduce(vcat, x_vecs)
    return x_vec, Vector_Tangent_from_vec
end

function to_vec_tangent(x::Array)
    x_vec, Tangent_from_vec = to_vec_tangent(vec(x))

    function Array_Tangent_from_vec(x_vec)
        return collect(reshape(Tangent_from_vec(x_vec), size(x)))
    end

    return x_vec, Array_Tangent_from_vec
end

function non_differentiable_type_to_vec_tangent(x)
    NoTangent_from_vec(v) = NoTangent()
    return Bool[], NoTangent_from_vec
end

to_vec_tangent(x::Char) = non_differentiable_type_to_vec_tangent(x)
to_vec_tangent(x::String) = non_differentiable_type_to_vec_tangent(x)
to_vec_tangent(x::Integer) = non_differentiable_type_to_vec_tangent(x)
to_vec_tangent(x::Complex{<:Integer}) = non_differentiable_type_to_vec_tangent(x)
to_vec_tangent(x::AbstractVector{<:Integer}) = non_differentiable_type_to_vec_tangent(x)

# Any struct ought to be interpretable as a Tangent, regardless inner constructors etc.
function to_vec_tangent(x::T) where {T}
    Base.isstructtype(T) || throw(error("Expected a struct type"))

    val_vecs_and_backs = [to_vec_tangent(getfield(x, name)) for name in fieldnames(T)]
    vals = first.(val_vecs_and_backs)
    backs = last.(val_vecs_and_backs)

    v, Tangents_from_vec = to_vec_tangent(vals)
    function structtype_Tangent_from_vec(v::Vector{<:Real})
        val_vecs = Tangents_from_vec(v)
        tangents = map((b, v) -> b(v), backs, val_vecs)
        if T <: Tuple
            return Tangent{T}(tangents...)
        else
            return Tangent{T}(NamedTuple(zip(fieldnames(T), tangents))...)
        end
    end
    return v, structtype_Tangent_from_vec
end

# Dictionaries are primitives, and must therefore have their own method.
function to_vec_tangent(d::T) where {T<:Dict}
    d_vec, tangent_from_vec = to_vec_tangent(collect(values(d)))
    function Dict_from_vec(v)
        v_vec_vec = tangent_from_vec(v)
        return Tangent{T}(Dict(key => v_vec_vec[n] for (n, key) in enumerate(keys(d))))
    end
    return d_vec, Dict_from_vec
end
