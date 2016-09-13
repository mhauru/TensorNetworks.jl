# To be included as a submodule of TensorNetworks
module TensorNetworkEvaluations

include("BinaryTrees.jl")
using .BinaryTrees
# From the parent module
# TODO Collapse the multiline imports, hopefully using ..:
import ..TensorNetwork, ..tensor, ..subnetwork, ..contractnodes!
import ..joinnetworks
import ..Bonds.getendpoints
import ..Nodes.combinelabels

export TensorNetworkEvaluation, tensor, evaluate!, subevaluation

# CTree is short for contraction tree.
typealias CTree BinaryTree{Symbol}
typealias CTreeNode BinaryNode{Symbol}

type TensorNetworkEvaluation
    tn::TensorNetwork
    tree::CTree
end

function TensorNetworkEvaluation(tn::TensorNetwork, bondorder::Vector)
    bondorder = collect(map(symbol, bondorder))
    # TODO the following should be unnecessary in 0.5.
    if length(bondorder) == 0
        bondorder = Vector{Symbol}()
    end
    tne = TensorNetworkEvaluation(tn, bondorder)
    return tne
end

function TensorNetworkEvaluation(tn::TensorNetwork, bondorder::Vector{Symbol})
    nodematrix = bondorder_to_nodematrix(tn, bondorder)
    if length(nodematrix) == 0
        # There's only one tensor in the network.
        nodelabel = first(keys(tn.nodes))
        tree = CTree(CTreeNode(nodelabel))
    else
        binarynodematrix = nodematrix_to_leafmatrix(nodematrix)
        tree = binarynodematrix_to_tree(binarynodematrix, combinelabels)
    end
    tne = TensorNetworkEvaluation(tn, tree)
    return tne
end

""" Turns a list of bonds representing a contraction order to matrix whose
first dimension corresponds to contractions and second dimension (of length 2)
lists the labels of the nodes contracted. This is an intermediate step in
turning this into a tree representation of the contraction order.
"""
function bondorder_to_nodematrix(tn::TensorNetwork, bondorder)
    if length(bondorder) == 0
        nodematrix = Array{Symbol,2}[]
    else
        nodevectors = Vector{Vector{Symbol}}()
        for i in 1:length(bondorder)
            bondlabel = bondorder[i]
            bond = tn.bonds[bondlabel]
            nodelabels = collect(getendpoints(bond))
            if length(nodelabels) > 1
                push!(nodevectors, nodelabels)
            end
        end
        nodematrix = transpose(hcat(nodevectors...))
        # Remove duplicates.
        nodematrix = unique(nodematrix, 1)
    end
    return nodematrix
end

function nodematrix_to_leafmatrix(nodematrix)
    nodelabels = unique(nodematrix)
    leafdict = Dict([l => CTreeNode(l) for l in nodelabels]...)
    leafmatrix = map(l -> leafdict[l], nodematrix)
    return leafmatrix
end

function binarynodematrix_to_tree(nodematrix, valuebuilder)
    nodes = nodematrix[1,:]
    newvalue = valuebuilder([n.value for n in nodes]...)
    newnode = CTreeNode(newvalue, nodes...)
    if size(nodematrix)[1] == 1
        tree = CTree(newnode)
    else
        nodematrix = nodematrix[2:end,:]
        # Note that the following relies on object identity.
        oldnodemask = map(x -> x in nodes, nodematrix)
        nodematrix[oldnodemask] = newnode
        # Remove duplicates.
        nodematrix = unique(nodematrix, 1)
        tree = binarynodematrix_to_tree(nodematrix, valuebuilder) 
    end
    return tree
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
        tn = contractnodes!(tn, root.value)
    else
        tnleft = evaluate!(subevaluation(tne, :left))
        tnright = evaluate!(subevaluation(tne, :right))
        tn = joinnetworks(tnleft, tnright)
        leftnode = first(keys(tnleft.nodes))
        rightnode = first(keys(tnright.nodes))
        contractnodes!(tn, leftnode, rightnode)
    end
    return tn
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
    root = tne.tree.root
    if direction === :left
        excludedbranch = get(root.right)
        includedbranch = get(root.left)
    else
        excludedbranch = get(root.left)
        includedbranch = get(root.right)
    end
    excludednodes = [node.value for node in leaves(excludedbranch)]
    subtn = subnetwork(tn, exclude=excludednodes)
    subtree = CTree(includedbranch)
    subeval = TensorNetworkEvaluation(subtn, subtree)
    return subeval
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Invariants

end
