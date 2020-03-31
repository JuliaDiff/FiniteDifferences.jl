"""
    to_vec(x)

Transform `x` into a `Vector`, and return the vector, and a closure which inverts the transformation.
"""
function to_vec(x::Number)
    function Number_from_vec(x_vec)
        return first(x_vec)
    end
    return [x], Number_from_vec
end

# Vectors
to_vec(x::Vector{<:Number}) = (x, identity)
function to_vec(x::Vector)
    x_vecs_and_backs = map(to_vec, x)
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    function Vector_from_vec(x_vec)
        sz = cumsum([map(length, x_vecs)...])
        return [backs[n](x_vec[sz[n]-length(x_vecs[n])+1:sz[n]]) for n in eachindex(x)]
    end
    return vcat(x_vecs...), Vector_from_vec
end

# Arrays
function to_vec(x::Array{<:Number})
    function Array_from_vec(x_vec)
        return reshape(x_vec, size(x))
    end
    return vec(x), Array_from_vec
end

function to_vec(x::Array)
    x_vec, back = to_vec(reshape(x, :))
    function Array_from_vec(x_vec)
        return reshape(back(x_vec), size(x))
    end
    return x_vec, Array_from_vec
end

# AbstractArrays
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
    function Transpose_from_vec(x_vec)
        return Transpose(permutedims(reshape(x_vec, size(X))))
    end
    return vec(Matrix(X)), Transpose_from_vec
end

function to_vec(X::Adjoint)
    function Adjoint_from_vec(x_vec)
        return Adjoint(conj!(permutedims(reshape(x_vec, size(X)))))
    end
    return vec(Matrix(X)), Adjoint_from_vec
end

# Non-array data structures

function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum([map(length, x_vecs)...])
    function Tuple_from_vec(v)
        return ntuple(n->x_backs[n](v[sz[n]-length(x_vecs[n])+1:sz[n]]), length(x))
    end
    return reduce(vcat, x_vecs), Tuple_from_vec
end

# Convert to a vector-of-vectors to make use of existing functionality.
function to_vec(d::Dict)
    d_vec_vec = [val for val in values(d)]
    d_vec, back = to_vec(d_vec_vec)
    function Dict_from_vec(v)
        v_vec_vec = back(v)
        return Dict([(key, v_vec_vec[n]) for (n, key) in enumerate(keys(d))])
    end
    return d_vec, Dict_from_vec
end
