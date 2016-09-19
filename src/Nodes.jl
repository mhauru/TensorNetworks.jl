
module Nodes

using TensorOperations
import Base.copy

# TODO invar
export Node, contractnodes, relabel!, bonddimension, copy, invar_tensororder

const blastypes = (Float32, Float64, Complex64, Complex128)    

type Node
    label::Symbol
    tensor::Array
    bonds::Vector{Symbol}
end

"""
Note that copying a Node creates a new node object with a new bond list,
but one that refers to the same underlying Array.
"""
function copy(node::Node)
    newnode = Node(node.label, node.tensor, copy(node.bonds))
    return newnode
end

function show(io::IO, tnn::Node)
    label = tnn.label
    bonds = tnn.bonds
    str = "Node $label, bonds: $a"
    return print(io, str)
end

function relabel!(node::Node, label)
    node.label = label
    return node
end

function combinelabels(label1)
    return label1
end

function combinelabels(label1, label2)
    str1, str2 = map(string, (label1, label2))
    newlabel = symbol("($str1,$str2)")
    return newlabel
end

function bondindex(node, label)
    indices = findin(node.bonds, [label])
    if length(indices) == 1
        indices = indices[1]
    end
    return indices
end

function bonddimension(node::Node, bondlabel)
    tensor = node.tensor
    inds = findin(node.bonds, [bondlabel])
    bonddimension = prod([size(tensor)[i] for i in inds])
    return bonddimension
end

function contractnodes(node, bondlabels)
    # Trace over a single node.
    tensor = node.tensor
    n = ndims(tensor)
    labels = collect(1:n)
    for s in bondlabels
        i1, i2 = bondindex(node, s)
        labels[i2] = labels[i1]
    end

    newtensor = tensortrace(tensor, labels)

    newlabel = node.label
    oldbondlabels = node.bonds
    newbondlabels = oldbondlabels[find(x -> !in(x, bondlabels), oldbondlabels)]
    newnode = Node(newlabel, newtensor, newbondlabels)
    return newnode
end

function contractnodes(node1, node2, bondlabels)
    # Two nodes contracted together.
    tensor1, tensor2 = node1.tensor, node2.tensor
    ndims1, ndims2 = ndims(tensor1), ndims(tensor2)
    labels1 = collect(1:ndims1)
    labels2 = collect(ndims1+1:ndims1+ndims2)
    for s in bondlabels
        i1 = bondindex(node1, s)
        i2 = bondindex(node2, s)
        labels2[i2] = labels1[i1]
    end

    # Check whether the element types of the tensors can be handled by BLAS.
    if eltype(tensor1) in blastypes && eltype(tensor2) in blastypes
        method = :BLAS
    else
        method = :native
    end
    newtensor = tensorcontract(tensor1, labels1, tensor2, labels2;
                               method=method)

    newlabel = combinelabels(node1.label, node2.label)
    newbondlabels = vcat(node1.bonds..., node2.bonds...)
    newbondlabels = newbondlabels[find(x -> !in(x, bondlabels),
                                       newbondlabels)]
    newnode = Node(newlabel, newtensor, newbondlabels)
    return newnode
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Invariants

function invar_tensororder(node::Node)
    if length(node.bonds) != ndims(node.tensor)
        errmsg = "Mismatch of tensor order in TensorNetwork Node:\n$node"
        throw(ArgumentError(errmsg))
    end
    return true
end

end

