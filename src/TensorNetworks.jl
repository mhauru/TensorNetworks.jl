module TensorNetworks

using TensorOperations
import Base.show
# TODO the following four lines should be unnecessary in julia v0.5
import Base.Symbol
import Base.convert
Symbol(x::Any) = symbol(x)
convert(::Type{Symbol}, x::Any) = symbol(x)

export TensorNetwork, addnode!, removenode!, addtensor!, relabelnode!,
       relabelbond!, tensor, contractbonds!, contractnodes!, subnetwork

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Submodules
# (each of these files includes one module, and one module only)

include("Bonds.jl")
include("Nodes.jl")
using .Bonds
using .Nodes


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Types

type TensorNetwork
    nodes::Dict{Symbol, Node}
    bonds::Dict{Symbol, Bond}
end

type TensorNetworkEvaluation
    tn::TensorNetwork
    order::Vector{Symbol}
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generating labels

function randomlabel(nodeorbond)
    s = randstring(3)
    if nodeorbond == :bond
        s = lowercase(s)
    elseif nodeorbond == :node
        s = uppercase(s)
    else
        errmsg = "Argument to randomlabel should be :node or :bond"
        throw(ArgumentError(errmsg))
    end
    s = symbol(s)
    return s
end

function createlabel(tn, nodeorbond)
    if nodeorbond == :bond
        labels = keys(tn.bonds)
    elseif nodeorbond == :node
        labels = keys(tn.nodes)
    else
        errmsg = "Argument to randomlabel should be :node or :bond"
        throw(ArgumentError(errmsg))
    end
    l = randomlabel(nodeorbond)
    while in(l, labels)
        # This should be good as long as the network as < 50k nodes.
        l = randomlabel(nodeorbond)
    end
    return l
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Constructors and such

function TensorNetwork()
    tn = TensorNetwork(Dict{Symbol, Node}(),
                       Dict{Symbol, Node}()) 
    return tn
end

function addnode!(tn, node)
    tn.nodes[node.label] = node
    allbonds = keys(tn.bonds)
    for bondlabel in node.bonds
        if in(bondlabel, allbonds) 
            # If the new node has a bond label that is already a bond in the
            # network, connect this (presumable dangling) bond to the new node.
            connectbond!(tn.bonds[bondlabel], node.label)
        else
            # Otherwise, create a new, dangling bond.
            tn.bonds[bondlabel] = Bond(bondlabel, node.label)
        end
    end
    return tn
end

function addtensor!(tn, tensor; label=nothing, bondlabels=nothing)
    if label == nothing
        label = createlabel(tn, :node)
    end
    if bondlabels==nothing
        bondlabels = [createlabel(tn, :bond) for i in 1:ndims(tensor)]
    end
    node = Node(label, tensor, bondlabels)
    addnode!(tn, node)
    return tn
end

function relabelnode!(tn, oldlabel, newlabel)
    node = tn.nodes[oldlabel]
    relabel!(node, newlabel)
    delete!(tn.nodes, oldlabel)
    tn.nodes[newlabel] = node
    for bondlabel in node.bonds
        bond = tn.bonds[bondlabel]
        reconnectbond!(bond, oldlabel, newlabel) 
    end
    return tn
end

function relabelbond!(tn, oldlabel, newlabel)
    bond = tn.bonds[oldlabel]
    relabel!(bond, newlabel)
    delete!(tn.bonds, oldlabel)
    tn.bonds[newlabel] = bond
    for nodelabel in getendpoints(bond)
        node = tn.nodes[nodelabel]
        bls = node.bonds
        bls[findin(bls, oldlabel)] = newlabel
    end
    return tn
end

function removenode!(tn, label)
    node = tn.nodes[label]
    delete!(tn.nodes, label)
    for s in node.bonds
        b = tn.bonds[s]
        disconnectbond!(b, node.label)
        if isdoubledangling(b)
            delete!(tn.bonds, s)
        end
    end
    return tn
end

function connectingbonds(tn, nodelabels...)
    nodes = [tn.nodes[l] for l in nodelabels]
    allmentionedbonds = vcat([n.bonds for n in nodes]...)
    # Get the ones that apper twice.
    once = Set{Symbol}()
    twice = Set{Symbol}()
    for l in allmentionedbonds
        if in(l, once)
            push!(twice, l)
        else
            push!(once, l)
        end
    end
    return twice
end

allnodelabels(tn) = Set{Symbol}(keys(tn.nodes))

function constructbondsfromnodes(nodedict)
    bonds = Dict{Symbol, Bond}
    for (nodelabel, node) in nodedict
        for bondlabel in node.bonds
            if bondlabel in keys(bonds)
                connectbond!(bonds[bondlabel], nodelabel)
            else
                bonds[bondlabel] = Bond(bondlabel, nodelabel)
            end
        end
    end
    return bonds
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Displaying

function Base.show(io::IO, tn::TensorNetwork)
    str = "TensorNetwork, with nodes\n"
    for (label, node) in tn.nodes
        str *= " $label: $(node.bonds)\n"
    end
    str *= "and bonds\n"
    for (label, bond) in tn.bonds
        ns = bond.nodes
        n1, n2 = length(ns) == 1 ? (first(ns), nothing) : ns
        str *= " $label: $n1 <=> $n2\n"
    end
    return print(io, str)
end

function Base.show(io::IO, tne::TensorNetworkEvaluation)
    str = "TensorNetworkEvaluation, with\n"*
          "$(tne.tn)\n"*
          "and contraction order:\n"*
          "$(tne.order)"
    return print(io, str)
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# High-level functions

function subnetwork(tn; include=allnodelabels(tn))
    allnodes = allnodelabels(tn)
    exclude = setdiff(allnodes, include)
    return subnetwork(tn, exclude=exclude)
end

function subnetwork(tn; exclude=Set{Symbol}())
    tn = copy(tn)  # TODO is this shallow?
    for l in exclude
        removenode!(tn, l)
    end
    return tn
end

function contract_backend!(tn, nodelabels, bondlabels)
    nodes = [tn.nodes[l] for l in nodelabels]
    newnode = contractnodes(nodes..., bondlabels)
    for l in nodelabels
        removenode!(tn, l)
    end
    addnode!(tn, newnode)
    return tn
end

function contractnodes!(tn, nodelabels...)
    bondlabels = connectingbonds(tn, nodelabels...)
    tn = contract_backend!(tn, nodelabels, bondlabels)
    return tn
end

function contractbonds!(tn, bondlabels...; allowpartial=false)
    nodelabels = union([Set{Symbol}(tn.bonds[l].nodes) for l in bondlabels]...)
    if !allowpartial
        # Make sure that all legs between nodes are contracted at once.
        contractnodes!(tn, nodelabels...)
    else
        # Allow for contracting only some legs connecting two tensors.
        contract_backend!(tn, nodelabels, bondlabels)
    end
    return tn
end

function tensor(x, outlabels::Vector)
    # TODO is this a good way to deal with outlabels consisting of something
    # other than symbols
    outlabels = map(symbol, outlabels)
    return tensor(x, outlabels)
end

function tensor(tne::TensorNetworkEvaluation, outlabels::Vector{Symbol})
    tn = tne.tn
    order = tne.order
    for l in order
        try
            contractbonds!(tn, l)
        catch KeyError
            # TODO This most probably means l had been contracted over in some
            # previous contraction. However, come up with a more robust way
            # of dealing with this.
        end
    end
    res = tensor(tn, outlabels)
    return res
end

function tensor(tn::TensorNetwork, outlabels::Vector{Symbol})
    if length(tn.nodes) > 1
        errmsg = "Can not extract a single tensor from a tensor network with"*
                 " more than one node. Use NCon to contract the network first."
        throw(ArgumentError(errmsg) )
    end
    node = first(tn.nodes).second
    tensor = node.tensor
    tensor = tensorcopy(tensor, node.bonds, outlabels)
    return tensor
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Invariants

# TODO should these return false or throw errors?

function invar_nodelabels(tn::TensorNetwork)
    for (label, node) in tn.nodes
        if label != node.label
            return false
        end
    end
    return true
end

function invar_bondlabels(tn::TensorNetwork)
    for (label, bond) in tn.bonds
        if label != bond.label
            return false
        end
    end
    return true
end

function invar_connectivity(tn::TensorNetwork)
    recobonds = constructbondsfromnodes(tn.nodes)
    # TODO How does object identity/equality work here?
    return recobonds == tn.bonds
end

function invar_tensorndims(tn::TensorNetwork)
    for node in values(tn.nodes)
        if length(node.bonds) != ndims(node.tensor)
            return false
        end
    end
    return true
end

function invar_indexcompatibility(tn::TensorNetwork)
    for (bondlabel, bond) in tn.bonds  
        nodes = bond.nodes
        if length(nodes) > 1
            # TODO This should be part of a tensor class.
            chi1, chi2 = [getbonddimension(tn.nodes[l], bondlabel)
                          for l in nodes]
            if chi1 != chi1
                return false
            end
        end
    end
    return true
end

function invar(tn::TensorNetwork)
    invars = []
    push!(invars, invar_nodelabels(tn))
    push!(invars, invar_bondlabels(tn))
    push!(invars, invar_connectivity(tn))
    push!(invars, invar_tensorndims(tn))
    push!(invars, invar_indexcompatibility(tn))
    return all(invars)
end

end
