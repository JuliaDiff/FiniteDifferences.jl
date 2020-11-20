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

function to_vec(x::AbstractVector)
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = cumsum(map(length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
        return oftype(x, x_Vec)
    end
    return vcat(x_vecs...), Vector_from_vec
end

function to_vec(x::AbstractArray)
    x_vec, from_vec = to_vec(vec(x))

    function Array_from_vec(x_vec)
        return oftype(x, reshape(from_vec(x_vec), size(x)))
    end

    return x_vec, Array_from_vec
end


# Some specific subtypes of AbstractArray.
function to_vec(x::Base.ReshapedArray{<:Any, 1})
    x_vec, from_vec = to_vec(parent(x))
    function ReshapedArray_from_vec(x_vec)
        p = from_vec(x_vec)
        return Base.ReshapedArray(p, x.dims, x.mi)
    end

    return x_vec, ReshapedArray_from_vec
end

# To return a SubArray we would endup needing to copy the `parent` of `x` in `from_vec`
# which doesn't seem particularly useful. So we just convert the view into a copy.
# we might be able to do something more performant but this seems good for now.
to_vec(x::Base.SubArray) = to_vec(copy(x))

function to_vec(x::T) where {T<:LinearAlgebra.AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    function AbstractTriangular_from_vec(x_vec)
        return T(reshape(back(x_vec), size(x)))
    end
    return x_vec, AbstractTriangular_from_vec
end

function to_vec(x::T) where {T<:LinearAlgebra.HermOrSym}
    x_vec, back = to_vec(Matrix(x))
    function HermOrSym_from_vec(x_vec)
        return T(back(x_vec), x.uplo)
    end
    return x_vec, HermOrSym_from_vec
end

function to_vec(X::Diagonal)
    x_vec, back = to_vec(Matrix(X))
    function Diagonal_from_vec(x_vec)
        return Diagonal(back(x_vec))
    end
    return x_vec, Diagonal_from_vec
end

function to_vec(X::Transpose)
    x_vec, back = to_vec(Matrix(X))
    function Transpose_from_vec(x_vec)
        return Transpose(permutedims(back(x_vec)))
    end
    return x_vec, Transpose_from_vec
end

function to_vec(X::Adjoint)
    x_vec, back = to_vec(Matrix(X))
    function Adjoint_from_vec(x_vec)
        return Adjoint(conj!(permutedims(back(x_vec))))
    end
    return x_vec, Adjoint_from_vec
end

function to_vec(X::PermutedDimsArray{T, D, P, Pinv} where {T, D}) where {P, Pinv}
    x_vec, back = to_vec(collect(X))
    function PermutedDimsArray_from_vec(x_vec)
        X_parent = collect(PermutedDimsArray(back(x_vec), Pinv))
        return PermutedDimsArray(X_parent, P)
    end
    return x_vec, PermutedDimsArray_from_vec
end

# Non-array data structures

function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum(collect(map(length, x_vecs)))
    function Tuple_from_vec(v)
        return ntuple(length(x)) do n
            return x_backs[n](v[sz[n] - length(x_vecs[n]) + 1:sz[n]])
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
function FiniteDifferences.to_vec(x::Composite{P}) where{P}
    x_canon = canonicalize(x)  # to be safe, fill in every field and put in primal order.
    x_inner = ChainRulesCore.backing(x_canon)
    x_vec, back_inner = FiniteDifferences.to_vec(x_inner)
    function Composite_from_vec(y_vec)
        y_back = back_inner(y_vec)
        return Composite{P, typeof(y_back)}(y_back)
    end
    return x_vec, Composite_from_vec
end


function FiniteDifferences.to_vec(x::AbstractZero)
    function AbstractZero_from_vec(x_vec::Vector)
        return x
    end
    return Bool[], AbstractZero_from_vec
end
