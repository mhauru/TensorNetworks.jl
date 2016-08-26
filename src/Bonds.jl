module Bonds

export Bond, connectbond!, disconnectbond!, reconnectbond!,
       isdangling, isdoubledangling, getendpoints

type Bond
    label::Symbol
    first::Nullable{Symbol}
    second::Nullable{Symbol}
end

function Bond(bondlabel, nodelabel1::Symbol, nodelabel2::Symbol)
    res = Bond(bondlabel, Nullable(nodelabel1),
                            Nullable(nodelabel2))  
    return res
end

function Bond(bondlabel, nodelabel::Symbol)
    res = Bond(bondlabel, Nullable(nodelabel), Nullable{Symbol}())  
    return res
end

function Base.show(io::IO, tnb::Bond)
    label = tnb.label
    nodelabel1, nodelabel2 = tnb.nodes
    str = "Bond $label: $nodelabel1 <=> $nodelabel2"
    return print(io, str)
end

function connectbond!(bond, nodelabel)
    if !isdangling(bond)
        errmsg = "Can not connect a bond that isn't dangling."
        throw(ArgumentError(errmsg))
    elseif isnull(bond.first)
        bond.first = Nullable(nodelabel)
    else
        bond.second = Nullable(nodelabel)
    end
    return bond
end

function relabel!(bond::Bond, label)
    bond.label = label
    return bond
end

function reconnectbond!(bond, oldlabel, newlabel)
    disconnectbond(bond, oldlabel)
    connectbond!(bond, newlabel)
    return bond
end

function disconnectbond!(bond, label)
    if bond.first == Nullable(label)
        bond.first = Nullable{Symbol}()
    elseif bond.second == Nullable(label)
        bond.second = Nullable{Symbol}()
    else
        errmsg = "Can not disconnect bond $(bond.label) from node $label,"*
                 " because it's not connected."
        raise(ArgumentError(errmsg))
    end
    return bond
end

function isdangling(bond::Bond)
    res = isnull(bond.first) || isnull(bond.second)
    return res
end

function isdoubledangling(bond)
    res = isnull(bond.first) && isnull(bond.second)
    return res
end

function getendpoints(bond)
    endpoints = Set{Symbol}()
    if !isnull(bond.first)
        push!(endpoints, get(bond.first))
    end
    if !isnull(bond.second)
        push!(endpoints, get(bond.second))
    end
    return endpoints
end

end

