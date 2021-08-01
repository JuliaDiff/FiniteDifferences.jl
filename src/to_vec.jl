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

# Integers cannot be perturbed!
function to_vec(x::Integer)
    Integer_from_vec(v) = x
    return Bool[], Integer_from_vec
end

# Base case -- if x is already a Vector{<:Real} there's no conversion necessary.
to_vec(x::Vector{<:Real}) = (x, identity)

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

# Fallback method for `to_vec`. Won't always do what you wanted, but should be fine a decent
# chunk of the time.
function to_vec(x::T) where {T}
    Base.isstructtype(T) || throw(error("Expected a struct type"))
    is_singleton(x) && return (Bool[], _ -> x) # Singleton types

    val_vecs_and_backs = get_val_vecs_and_backs(x)
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

# Type-stable way to determine whether a type has any fields.
@generated function is_singleton(x)
    return isempty(fieldnames(x)) ? :true : :false
end

# Type-stable way to call `to_vec` on each field.
@generated function get_val_vecs_and_backs(x)
    return Expr(:tuple, map(name -> :(to_vec(x.$name)), fieldnames(x))...)
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

# To return a SubArray we would endup needing to copy the `parent` of `x` in `from_vec`
# which doesn't seem particularly useful. So we just convert the view into a copy.
# we might be able to do something more performant but this seems good for now.
to_vec(x::Base.SubArray) = to_vec(copy(x))

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

function to_vec(x::Tuple{})
    Tuple_from_vec(v) = ()
    return Bool[], Tuple_from_vec
end

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


# ChainRulesCore Differentials
function FiniteDifferences.to_vec(x::Tangent{P}) where{P}
    x_canon = canonicalize(x)  # to be safe, fill in every field and put in primal order.
    x_inner = ChainRulesCore.backing(x_canon)
    x_vec, back_inner = FiniteDifferences.to_vec(x_inner)
    function Tangent_from_vec(y_vec)
        y_back = back_inner(y_vec)
        return Tangent{P, typeof(y_back)}(y_back)
    end
    return x_vec, Tangent_from_vec
end

function FiniteDifferences.to_vec(x::AbstractZero)
    function AbstractZero_from_vec(x_vec::Vector)
        return x
    end
    return Bool[], AbstractZero_from_vec
end

function FiniteDifferences.to_vec(t::Thunk)
    v, back = to_vec(unthunk(t))
    Thunk_from_vec = v -> @thunk(back(v))
    return v, Thunk_from_vec
end

# Things that aren't struct types and aren't differentiable.
to_vec(x::Char) = Bool[], _ -> x
to_vec(x::Symbol) = Bool[], _ -> x
