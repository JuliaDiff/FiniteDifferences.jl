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

to_vec(x::Char) = (Bool[], _ -> x)

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
    isempty(fieldnames(T)) && return (Bool[], _ -> x) # Singleton types

    val_vecs_and_backs = map(name -> to_vec(getfield(x, name)), fieldnames(T))
    vals = first.(val_vecs_and_backs)
    backs = last.(val_vecs_and_backs)

    v, vals_from_vec = to_vec(vals)
    function structtype_from_vec(v::Vector{<:Real})
        val_vecs = vals_from_vec(v)
        values = map((b, v) -> b(v), backs, val_vecs)
        try
            T(values...)
        catch MethodError
            return _force_construct(T, values...)
        end
    end
    return v, structtype_from_vec
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

function to_vec(x::F) where {F <: SVD}
    # Convert the vector S to a matrix so we can work with a vector of matrices
    # only and inferrence work
    v = [x.U, reshape(x.S, length(x.S), 1), x.Vt]
    x_vec, back = to_vec(v)
    function SVD_from_vec(v)
        U, Smat, Vt = back(v)
        return F(U, vec(Smat), Vt)
    end
    return x_vec, SVD_from_vec
end

function to_vec(x::Cholesky)
    x_vec, back = to_vec(x.factors)
    function Cholesky_from_vec(v)
        return Cholesky(back(v), x.uplo, x.info)
    end
    return x_vec, Cholesky_from_vec
end

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

# Non-array data structures

function to_vec(x::Tuple)
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = typeof(lengths)(cumsum(collect(lengths)))
    function Tuple_from_vec(v)
        map(x_backs, lengths, sz) do x_back, l, s
            return x_back(v[s - l + 1:s])
        end
    end
    return reduce(vcat, x_vecs), Tuple_from_vec
end

function to_vec(x::NamedTuple)
    x_vec, back = to_vec(values(x))
    function NamedTuple_from_vec(v)
        v_vec_vec = back(v)
        return typeof(x)(v_vec_vec)
    end
    return x_vec, NamedTuple_from_vec
end

# Convert to a vector-of-vectors to make use of existing functionality.
function to_vec(d::Dict)
    d_vec, back = to_vec(collect(values(d)))
    function Dict_from_vec(v)
        v_vec_vec = back(v)
        return Dict(key => v_vec_vec[n] for (n, key) in enumerate(keys(d)))
    end
    return d_vec, Dict_from_vec
end


# NOTE: to_vec has some problems at the minute, so the additional code below is code that
# needs to be written regardless what we do with the rest of the ecosystem.

"""
    to_vec_Tangent(x)

Transform `x` into a `Vector`, and return the vector, and a closure which builds a
`Tangent`.
"""
function to_vec_Tangent(x::Real)
    Real_Tangent_from_vec(x_vec) = first(x_vec)
    return [x], Real_Tangent_from_vec
end

function to_vec_Tangent(z::Complex)
    Complex_Tangent_from_vec(z_vec) = Complex(z_vec[1], z_vec[2])
    return [real(z), imag(z)], Complex_Tangent_from_vec
end

# Base case -- if x is already a Vector{<:Real} there's no conversion necessary.
to_vec_Tangent(x::Vector{<:Real}) = (x, identity)

to_vec_Tangent(x::Char) = (Bool[], _ -> x)

# Any struct ought to be interpretable as a Tangent, regardless inner constructors etc.
function to_vec_Tangent(x::T) where {T}
    Base.isstructtype(T) || throw(error("Expected a struct type"))
    isempty(fieldnames(T)) && return (Bool[], _ -> x) # Singleton types

    val_vecs_and_backs = map(name -> to_vec_Tangent(getfield(x, name)), fieldnames(T))
    vals = first.(val_vecs_and_backs)
    backs = last.(val_vecs_and_backs)

    v, Tangents_from_vec = to_vec_Tangent(vals)
    function structtype_Tangent_from_vec(v::Vector{<:Real})
        val_vecs = Tangents_from_vec(v)
        tangents = map((b, v) -> b(v), backs, val_vecs)
        return Tangent{T}(NamedTuple(zip(fieldnames(T), tangents))...)
    end
    return v, structtype_Tangent_from_vec
end

interpret_as_Tangent(x::Array) = map(interpret_as_Tangent(x))

function to_vec_Tangent(x::Vector)
    x_vecs_and_backs = map(to_vec_Tangent, x)
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

function to_vec_Tangent(x::Array)
    x_vec, Tangent_from_vec = to_vec_Tangent(vec(x))

    function Array_Tangent_from_vec(x_vec)
        return collect(reshape(Tangent_from_vec(x_vec), size(x)))
    end

    return x_vec, Array_Tangent_from_vec
end


# Going to need to do this in order to make equality work properly.
canonicalise_junk(x::Real) = x

canonicalise_junk(x::Complex) = x

canonicalise_junk(x::DenseArray{<:Real}) = x

canonicalise_junk(x::DenseArray) = map(canonicalise_junk, x)

canonicalise_junk(x::StridedArray{<:Real}) = x

canonicalise_junk(x::StridedArray) = map(canonicalise_junk, x)

canonicalise_junk(x::Char) = x

function canonicalise_junk(x::Tangent{<:Symmetric{<:Real, <:DenseArray{<:Real}}})

end

# It's unclear what the generic solution is here. Whether a generic canonicalisation
# operation exists is unclear, and whether 
canonicalise_junk(x::Tangent{<:Symmetric})

# Is there a solution which might circumvent this problem entirely? For example,
# pre-composing a function which needs testing


# Is this a problem at all in practice if we take structural tangents seriously?
# How is it that one can arrive 

# Assume that a programme can output e.g. two UpperTriangulars with arbitrary lower
# triangles. How is it possible to compare them? Is there an operation on which we can rely
# that gives us a canonical way to compare the data structures?
# Unclear what the correct solution is here...

# 1. is this only a problem with finite differencing?
# 2. would defensive mutation of the primals make sense here?
# 3. is there a way to create defensive properties by modifying the forwards pass? e.g.
#   by making `Array{T, N}(undef, size)` be filled with a canonical value of some kind?

# What if the type author also provides a function called `remove_junk`, or something like
# that, which promises to return only the bits of the data type which are not junk?
# `collect` would usually be fine for `AbstractArray`s (I think).

# The basic goal of these operations would be to ensure that a plain struct-like data
# structure is returned from any given function.
# The default could be something like
