module TensorNetworks

export TensorNetwork, addnode!, removenode!, addtensor!, relabelnode!,
       relabelbond!, tensor, contractbonds!, contractnodes!, subnetwork,
       joinnetworks, copy, ncon

using TensorOperations
import Base: show, copy
# TODO the following four lines should be unnecessary in julia v0.5
import Base.Symbol
import Base.convert
Symbol(x::Any) = symbol(x)
convert(::Type{Symbol}, x::Any) = symbol(x)


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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Constructors and such

TensorNetwork() = TensorNetwork(Dict{Symbol, Node}(), Dict{Symbol, Bond}()) 

function TensorNetwork(tensors, bondlists)
    # TODO This is a bit hacky, and relies on nothing being the default value
    # for label in addtensor!.
    labels = fill(nothing, length(tensors))
    tn = TensorNetwork(tensors, labels, bondlists)
    invar(tn)  # TODO invar 
    return tn
end

function TensorNetwork(tensors, labels, bondlists)
    tn = TensorNetwork()
    for (tensor, label, bondlabels) in zip(tensors, labels, bondlists)
        addtensor!(tn, tensor; label=label, bondlabels=bondlabels)
    end
    invar(tn)  # TODO invar 
    return tn
end

function copy(tn::TensorNetwork)
    newnodes = Dict{Symbol, Node}()
    for (nodelabel, node) in tn.nodes
        newnodes[nodelabel] = copy(node)
    end
    newbonds = Dict{Symbol, Bond}()
    for (bondlabel, bond) in tn.bonds
        newbonds[bondlabel] = copy(bond)
    end
    newtn = TensorNetwork(newnodes, newbonds)
    invar(tn)  # TODO invar
    return newtn
end

function addnode!(tn::TensorNetwork, node)
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
    invar(tn)  # TODO invar 
    return tn
end

function addtensor!(tn::TensorNetwork, tensor; label=nothing, bondlabels=nothing)
    if label == nothing
        label = createlabel(tn, :node)
    end
    if bondlabels==nothing
        bondlabels = [createlabel(tn, :bond) for i in 1:ndims(tensor)]
    end
    node = Node(label, tensor, bondlabels)
    addnode!(tn, node)
    invar(tn)  # TODO invar 
    return tn
end

function relabelnode!(tn::TensorNetwork, oldlabel, newlabel)
    node = tn.nodes[oldlabel]
    relabel!(node, newlabel)
    delete!(tn.nodes, oldlabel)
    tn.nodes[newlabel] = node
    for bondlabel in node.bonds
        bond = tn.bonds[bondlabel]
        reconnectbond!(bond, oldlabel, newlabel) 
    end
    invar(tn)  # TODO invar 
    return tn
end

function relabelbond!(tn::TensorNetwork, oldlabel, newlabel)
    bond = tn.bonds[oldlabel]
    relabel!(bond, newlabel)
    delete!(tn.bonds, oldlabel)
    tn.bonds[newlabel] = bond
    for nodelabel in getendpoints(bond)
        node = tn.nodes[nodelabel]
        bls = node.bonds
        bls[findin(bls, oldlabel)] = newlabel
    end
    invar(tn)  # TODO invar 
    return tn
end

function removenode!(tn::TensorNetwork, label)
    node = tn.nodes[label]
    delete!(tn.nodes, label)
    for s in node.bonds
        b = tn.bonds[s]
        disconnectbond!(b, node.label)
        if isdoubledangling(b)
            delete!(tn.bonds, s)
        end
    end
    invar(tn)  # TODO invar 
    return tn
end

function replacenodes!(tn::TensorNetwork, oldnodelabels, newnodes)
    for l in oldnodelabels
        removenode!(tn, l)
    end
    for n in newnodes
        addnode!(tn, n)
    end
    invar(tn)  # TODO invar 
    return tn
end

function connectingbonds(tn::TensorNetwork, nodelabels...; includetraces=true)
    nodes = [tn.nodes[l] for l in nodelabels]
    bondlists = [n.bonds for n in nodes]
    if !includetraces
        bondlists = map(unique, bondlists)
    end
    allmentionedbonds = vcat(bondlists...)
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

function joinnetworks(tn1::TensorNetwork, tn2::TensorNetwork)
    tn = copy(tn1)
    for node in values(tn2.nodes)
        addnode!(tn, node)
    end
    invar(tn)  # TODO invar
    return tn
end

allnodelabels(tn) = Set{Symbol}(keys(tn.nodes))

function constructbondsfromnodes(nodedict)
    bonds = Dict{Symbol, Bond}()
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
# Displaying

function Base.show(io::IO, tn::TensorNetwork)
    str = "TensorNetwork, with nodes\n"
    for (label, node) in tn.nodes
        str *= " $label: $(string(node.bonds))\n"
    end
    str *= "and bonds"
    for (label, bond) in tn.bonds
        str *= "\n $label: "
        n1, n2 = get(bond.first, :FREE), get(bond.second, :FREE)
        if n1 === :FREE
            str *= "$n2"
        elseif n2 === :FREE
            str *= "$n1"
        else
            str *= "$n1 <=> $n2"
        end
    end
    return print(io, str)
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# High-level functions

function subnetwork(tn::TensorNetwork; include=allnodelabels(tn))
    allnodes = allnodelabels(tn)
    exclude = setdiff(allnodes, include)
    return subnetwork(tn, exclude=exclude)
end

function subnetwork(tn::TensorNetwork; exclude=Set{Symbol}())
    tn = copy(tn)
    for l in exclude
        removenode!(tn, l)
    end
    invar(tn)  # TODO invar
    return tn
end

function contract_backend!(tn::TensorNetwork, nodelabels, bondlabels)
    nodes = [tn.nodes[l] for l in nodelabels]
    newnode = contractnodes(nodes..., bondlabels)
    replacenodes!(tn, nodelabels, [newnode])
    invar(tn)  # TODO invar
    return tn
end

function contractnodes!(tn::TensorNetwork, nodelabels...)
    # Note that, even if some of the nodes have tracing indices, those will
    # not be contracted over, unless that's the only node specified.
    trace = length(nodelabels) == 1
    bondlabels = connectingbonds(tn, nodelabels...; includetraces=trace)
    tn = contract_backend!(tn, nodelabels, bondlabels)
    invar(tn)  # TODO invar
    return tn
end

function contractbonds!(tn::TensorNetwork, bondlabels...; allowpartial=false)
    nodelabels = union([getendpoints(tn.bonds[l]) for l in bondlabels]...)
    if !allowpartial
        # Make sure that all legs between nodes are contracted at once.
        contractnodes!(tn, nodelabels...)
    else
        # Allow for contracting only some legs connecting two tensors.
        contract_backend!(tn, nodelabels, bondlabels)
    end
    invar(tn)  # TODO invar
    return tn
end

function tensor(t::TensorNetwork, outlabels::Vector)
    # TODO is this a good way to deal with outlabels consisting of something
    # other than symbols?
    outlabels = map(symbol, outlabels)
    # TODO The following work around should be unnecessary in 0.5.
    if length(outlabels) == 0
        outlabels = Vector{Symbol}()
    end
    return tensor(t, outlabels)
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

function invar_nodelabels(tn::TensorNetwork)
    for (label, node) in tn.nodes
        if label != node.label
            errmsg = "Mismatch of node labels in TensorNetwork:\n$tn"
            throw(ArgumentError(errmsg))
        end
    end
    return true
end

function invar_bondlabels(tn::TensorNetwork)
    for (label, bond) in tn.bonds
        if label != bond.label
            errmsg = "Mismatch of bond labels in TensorNetwork:\n$tn"
            throw(ArgumentError(errmsg))
        end
    end
    return true
end

function invar_connectivity(tn::TensorNetwork)
    recobonds = constructbondsfromnodes(tn.nodes)
    if recobonds != tn.bonds
        errmsg = "Mismatch of network connectivity from nodes and from bonds"*
                 " in TensorNetwork\n$tn"
        throw(ArgumentError(errmsg))
    end
    return true
end

function invar_tensororders(tn::TensorNetwork)
    nodeinvars = Vector{Bool}()
    for node in values(tn.nodes)
        push!(nodeinvars, invar_tensororder(node))
    end
    return all(nodeinvars)
end

function invar_indexcompatibility(tn::TensorNetwork)
    for (bondlabel, bond) in tn.bonds  
        nodes = getendpoints(bond)
        if length(nodes) > 1
            chi1, chi2 = [getbonddimension(tn.nodes[l], bondlabel)
                          for l in nodes]
            if chi1 != chi2
                errmsg = "Mismatch of bond dimensions for bond $bondlabel"*
                         " in TensorNetwork:\n$tn"
                throw(ArgumentError(errmsg))
            end
        end
    end
    return true
end

function invar(tn::TensorNetwork)
    invars = Vector{Bool}()
    push!(invars, invar_nodelabels(tn))
    push!(invars, invar_bondlabels(tn))
    push!(invars, invar_connectivity(tn))
    push!(invars, invar_tensororders(tn))
    push!(invars, invar_indexcompatibility(tn))
    return all(invars)
end


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Submodules with additional functionality
# (Each file contains one module, and one module only.)

include("TensorNetworkEvaluations.jl")
include("NCon.jl")
importall .NCon
importall .TensorNetworkEvaluations

end

