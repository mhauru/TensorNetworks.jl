# TODO Make use of AbstractTrees when we hit 0.5.

module BinaryTrees
export BinaryTree, isleaf, children, descendants, leaves, cutbranch!

type Node{T}
    value::T
    left::Nullable{Node{T}}
    right::Nullable{Node{T}}
end

type BinaryTree{T}
    root::Node{T}
end

function isleaf{T}(n::Node{T})
    return isnull(n.left) && isnull(n.right)
end

function children{T}(n::Node{T})
    cs = Vector{Node{T}}
    if !isnull(n.left)
        push!(cs, get(n.left))
    end
    if !isnull(n.right)
        push!(cs, get(n.right))
    end
    return cs
end

function descendants{T}(n::Node{T})
    descs = Vector{Node{T}}
    cs = children(n)
    push!(descs, cs...)
    for c in cs
        push!(descs, descendants(c))
    end
    return descs
end

function leaves{T}(n::Node{T})
    if isleaf(n)
        ls = [n]
    else
        cs = children(n)
        ls = vcat(map(leaves, cs))
    end
    return ls
end

function cutbranch!{T}(n::Node{T}, which::Symbol)
    if !(which in (:left, :right))
        errmsg = "Second argument of cutbranch needs to be :left or :right."
        throw(ArgumentError(errmsg))
    end
    if which === :left
        n.left = Nullable{typeof(n)}()
    else
        n.right = Nullable{typeof(n)}()
    end
    return n
end

end

