# To be included as a submodule of TensorNetworks
module TensorNetworkEvaluations

include("BinaryTrees.jl")
using .BinaryTrees
import ..TensorNetwork  # From the parent module

export TensorNetworkEvaluation, tensor, evaluate!, subevaluation

typealias NodeTree BinaryTree{Symbol}

type TensorNetworkEvaluation
    tn::TensorNetwork
    tree::NodeTree
end

function Base.show(io::IO, tne::TensorNetworkEvaluation)
    # TODO Print the tree using AbstractTrees
    bondorder = treetobondorder(tne.tree)
    str = "TensorNetworkEvaluation, with\n"*
          "$(tne.tn)\n"*
          "and contraction order:\n"*
          "$(tne.order)"
    return print(io, str)
end

function evaluate!(tne::TensorNetworkEvaluation)
    tn = tne.tn
    root = tne.tree.root
    if isleaf(root)
        # This takes care of possible traces.
        contractnodes!(tn, root.value)
    else
        evaluate!(subevaluation(tne, :left))
        evaluate!(subevaluation(tne, :right))
        leftnode = root.left.value
        rightnode = root.right.value
        contractnodes!(tn, leftnode, rightnode)
        cutbranch!(root, :left)
        cutbranch!(root, :right)
    end
    return tne.tn
end

# TODO This is doubled from TensorNetworks, decide where to put it.
function tensor(t::TensorNetworkEvaluation, outlabels::Vector)
    # TODO is this a good way to deal with outlabels consisting of something
    # other than symbols?
    outlabels = map(symbol, outlabels)
    # TODO The following work around should be unnecessary in 0.5.
    if length(outlabels) == 0
        outlabels = Vector{Symbol}()
    end
    return tensor(t, outlabels)
end

function tensor(tne::TensorNetworkEvaluation, outlabels::Vector{Symbol})
    tn = evaluate!(tne)
    t = tensor(tn, outlabels)
    return t
end

function subevaluation(tne::TensorNetworkEvaluation, direction::Symbol)
    if !(direction in (:left, :right))
        errmsg = "direction needs to be :left or :right."
        throw(ArgumentError(errmsg))
    end
    tn = tne.tn
    tree = tne.tree
    excludedbranch = direction === :left? tree.right : tree.left
    includedbranch = direction === :left? tree.left : tree.right
    excludednodes = map(value, leaves(excludednodes))
    subtn = subnetwork(tn, excludednodes)
    subeval = TensorNetworkEvaluation(subtn, includedbranch)
    return subeval
end

end

