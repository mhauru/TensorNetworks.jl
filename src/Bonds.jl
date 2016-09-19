module Bonds

import Base: ==, copy

export Bond, connectbond!, disconnectbond!, reconnectbond!,
       isdangling, isdoubledangling, endpoints, relabel!

type Bond
    label::Symbol
    first::Nullable{Symbol}
    second::Nullable{Symbol}
end

function nullorequal(a::Nullable{Symbol}, b::Nullable{Symbol})
    if isnull(a) && isnull(b)
        return true
    end
    if isnull(a) || isnull(b) 
        return false
    end
    return get(a) == get(b)
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

function copy(bond::Bond)
    newbond = Bond(bond.label, bond.first, bond.second)
    return newbond
end

function show(io::IO, bond::Bond)
    label = bond.label
    end1 = isnull(bond.first) ? "DANGLING" : get(bond.first)
    end2 = isnull(bond.second) ? "DANGLING" : get(bond.second)
    str = "Bond $label: $end1 <=> $end2"
    return print(io, str)
end

function ==(b1::Bond, b2::Bond)
    if b1.label !== b2.label
        return false
    end
    # TODO I feel like this isn't the most elegant way to go about doing this.
    if ((nullorequal(b1.first, b2.first) && nullorequal(b1.second, b2.second))
        ||
        (nullorequal(b1.first, b2.second) && nullorequal(b1.second, b2.first)))
        return true
    end
    return false
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
    disconnectbond!(bond, oldlabel)
    connectbond!(bond, newlabel)
    return bond
end

function disconnectbond!(bond, label)
    if !isnull(bond.first) && get(bond.first) == label
        bond.first = Nullable{Symbol}()
    elseif !isnull(bond.second) && get(bond.second) == label
        bond.second = Nullable{Symbol}()
    else
        errmsg = "Can not disconnect bond $(bond.label) from node $label,"*
                 " because it's not connected."
        throw(ArgumentError(errmsg))
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

function endpoints(bond)
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

