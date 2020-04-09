"""
    to_vec(x)

Transform `x` into a `Vector`, and return the vector, and a closure which inverts the
transformation.
"""
function to_vec(x::T) where {T<:Number}
    function Number_from_vec(x_vec)
        return T(first(x_vec))
    end
    return [x], Number_from_vec
end

# (Abstract)Vectors
to_vec(x::Vector{<:Number}) = (x, identity)

function to_vec(x::T) where {T<:AbstractVector}
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = cumsum(map(length, x_vecs))
        x_Vec = [backs[n](x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
        return T(x_Vec)
    end
    return vcat(x_vecs...), Vector_from_vec
end

# (Abstract)Arrays
function to_vec(x::T) where {T<:AbstractArray}

    x_vec, from_vec = to_vec(vec(x))

    function Array_from_vec(x_vec)
        return T(reshape(from_vec(x_vec), size(x)))
    end

    return x_vec, Array_from_vec
end

# Some specific subtypes of AbstractArray.

function to_vec(x::T) where {T<:LinearAlgebra.AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    function AbstractTriangular_from_vec(x_vec)
        return T(reshape(back(x_vec), size(x)))
    end
    return x_vec, AbstractTriangular_from_vec
end

function to_vec(x::Symmetric)
    function Symmetric_from_vec(x_vec)
        return Symmetric(reshape(x_vec, size(x)))
    end
    return vec(Matrix(x)), Symmetric_from_vec
end

function to_vec(X::Diagonal)
    function Diagonal_from_vec(x_vec)
        return Diagonal(reshape(x_vec, size(X)...))
    end
    return vec(Matrix(X)), Diagonal_from_vec
end

function to_vec(X::Transpose)

    x_vec, x_from_vec = to_vec(X.parent)

    function Transpose_from_vec(x_vec)
        return Transpose(x_from_vec(x_vec))
    end

    return x_vec, Transpose_from_vec
end

function to_vec(X::Adjoint)

    x_vec, x_from_vec = to_vec(X.parent)

    function Adjoint_from_vec(x_vec)
        return Adjoint(x_from_vec(x_vec))
    end

    return x_vec, Adjoint_from_vec
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

# Convert to a vector-of-vectors to make use of existing functionality.
function to_vec(d::Dict)
    d_vec, back = to_vec(collect(values(d)))
    function Dict_from_vec(v)
        v_vec_vec = back(v)
        return Dict(key => v_vec_vec[n] for (n, key) in enumerate(keys(d)))
    end
    return d_vec, Dict_from_vec
end
