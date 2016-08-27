
module Nodes

export Node, contractnodes, relabel!

const blastypes = (Float32, Float64, Complex64, Complex128)    

type Node
    label::Symbol
    tensor::Array
    bonds::Vector{Symbol}
end

function Base.show(io::IO, tnn::Node)
    label = tnn.label
    bonds = tnn.bonds
    str = "Node $label, bonds: $a"
    return print(io, str)
end

function relabel!(node::Node, label)
    node.label = label
    return node
end

function getbondindex(node, label)
    indices = findin(node.bonds, [label])
    if length(indices) == 1
        indices = indices[1]
    end
    return indices
end

function getbonddimension(node::Node, bondlabel)
    tensor = node.tensor
    i = findin(node.bonds, bondlabel)
    bonddimension = size(tensor)[i]
    return bonddimension
end

function contractnodes(node, bondlabels)
    # Trace over a single node.
    tensor = node.tensor
    ndims = ndims(tensor)
    labels = collect(1:ndims)
    for s in bondlabels
        i1, i2 = getbondindex(node, s)
        labels[i2] = labels[i1]
    end

    newtensor = tensortrace(tensor, labels)

    newlabel = node.label
    oldbondlabels = node.bondlabels
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
        i1 = getbondindex(node1, s)
        i2 = getbondindex(node2, s)
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

    str1, str2 = string(node1.label), string(node2.label)
    newlabel = "($str1,$str2)"
    newbondlabels = vcat(node1.bonds..., node2.bonds...)
    newbondlabels = newbondlabels[find(x -> !in(x, bondlabels),
                                       newbondlabels)]
    newnode = Node(newlabel, newtensor, newbondlabels)
    return newnode
end

end

