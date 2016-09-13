# To be included as a submodule of TensorNetworks
module NCon
export ncon
using ..TensorNetworkEvaluations
using ..TensorNetworks

"""
    ncon(L, v; forder=nothing, order=nothing, check_indices=false)

Takes a tuple of tensors L and a tuple of Array[Integer,1]s v that specifies
how these tensors form a network, and contracts this network. The return value
is the tensor that is formed as the result of this contraction.

More specifically:
L = (A1, A2, ..., Ap) is a tuple of tensors or a single tensor.

v = (v1, v2, ..., vp) is a tuple of Arrays of indices, each of which
corresponds to one of the tensors in L. For instance, if v1 = (3,4,-1) and 
v2 = (-2,-3,4), then the second index of A1 and the last index of A2 are
contracted together, because they are both labeled by 4. All positive numbers
label indices that are to be contracted and all negative numbers label indices
that are to remain open. The open indices will be the remaining indices of the
tensor that is formed. The index lists v1, v2, etc. may also be tuples instead
of Arrays, and v may consist of a single index list v1, if there's only one
tensor in the contraction (a trace).

order, if present, contains a tuple or Array of all positive indices - if not
(1, 2, 3, 4, ...) by default. This is the order in which the contractions are
performed. However, whenever an index joining two tensors is about to be
contracted together, ncon contracts at the same time all indices connecting
these two tensors, even if some of them only come up later in order.

forder, if present, contains the final ordering of the uncontracted indices
- if not, (-1, -2, ...) by default.

If check_indices is true (by default it's false) then checks are performed to
make sure the contraction is well-defined. If not, an ArgumentError with a
helpful description of what went wrong is provided.
"""
function ncon(L, v; forder=nothing, order=nothing, check_indices=true)
    # We want to handle the tensors as an Array{AbstractArray, 1}, instead of a
    # tuple. In addition, if only a single element is given, we make an Array
    # out of it. Inputs are assumed to be non-empty.
    if isa(L, AbstractArray) && eltype(L) <: Number
        # L is a single Tensor
        L = AbstractArray[L]
    else
        # L is not an Array, so let's make it one.
        L = AbstractArray[L...]
    end
    # The same thing for v, which we want to be of type Array{Array{Int,1}}.
    if isa(v[1], Number)
        # v is an index list for just one tensor, so wrap it in an Array.
        v = Array{Int,1}[collect(v)]
    end
    v = Array{Int,1}[[i...] for i in v]

    if order == nothing
        order = create_order(v)
    end
    if forder == nothing
        forder = create_forder(v)
    end

    tn = TensorNetwork(L, v)
    tne = TensorNetworkEvaluation(tn, order)

    result = tensor(tne, forder)
    return result
end


""" Identify all unique, positive indices and return them sorted. """
function create_order(v)
    order = sort(filter!(x -> x > 0, unique(vcat(v...))))
    return order
end


"""
Identify all unique, negative indices and return them reverse sorted (-1
first).
"""
function create_forder(v)
    forder = sort(filter!(x -> x < 0, unique(vcat(v...))), rev=true)
    return forder
end

end
