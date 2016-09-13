# TODO Make use of AbstractTrees when we hit 0.5.

module BinaryTrees
export BinaryTree, BinaryNode, isleaf, children, descendants, leaves,
       cutbranch!

type BinaryNode{T}
    value::T
    left::Nullable{BinaryNode{T}}
    right::Nullable{BinaryNode{T}}

    BinaryNode(value, left, right) = new(value, left, right)

    """ Creates a leaf. """
    function BinaryNode(value::T)
        null = Nullable{BinaryNode{T}}()
        node = BinaryNode{T}(value, null, null)
        return node 
    end
end


type BinaryTree{T}
    root::BinaryNode{T}

    function BinaryTree(root::BinaryNode{T})
        tree = new(root)
        invariant_istree(tree)
        return tree
    end
end

function isleaf{T}(n::BinaryNode{T})
    return isnull(n.left) && isnull(n.right)
end

function children{T}(n::BinaryNode{T})
    cs = Vector{BinaryNode{T}}()
    if !isnull(n.left)
        push!(cs, get(n.left))
    end
    if !isnull(n.right)
        push!(cs, get(n.right))
    end
    return cs
end

function descendants{T}(n::BinaryNode{T})
    descs = Vector{BinaryNode{T}}()
    cs = children(n)
    push!(descs, cs...)
    for c in cs
        push!(descs, descendants(c))
    end
    return descs
end

function leaves{T}(n::BinaryNode{T})
    if isleaf(n)
        ls = [n]
    else
        cs = children(n)
        ls = vcat(map(leaves, cs)...)
    end
    return ls
end

function cutbranch!{T}(n::BinaryNode{T}, which::Symbol)
    if !(which in (:left, :right))
        # TODO should this invariant checking happen everytime?
        # Should it happen elsewhere too?
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Invariants

function invariant_istree{T}(tree::BinaryTree{T})
    return invariant_istree(tree.root) 
end

function invariant_istree{T}(root::BinaryNode{T}, nodeset=Set{BinaryNode{T}}())
    if root in nodeset
        errmsg = "Tree is cyclic."
        # TODO What's the right error type here?
        throw(ArgumentError(errmsg))
    end
    push!(nodeset, root)
    left, right = root.left, root.right
    leftinvar = isnull(left) ? true : invariant_istree(get(left), nodeset)
    rightinvar = isnull(right) ? true : invariant_istree(get(right), nodeset)
    return leftinvar && rightinvar
end

end
