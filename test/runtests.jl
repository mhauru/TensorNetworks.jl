using TensorNetworks
using TensorNetworks.Bonds
using TensorNetworks.Nodes
using Base.Test
using NCon
using TensorOperations


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Tests for Bonds

b = Bond(:a, :A, :B)
@test !isdangling(b)
@test !isdoubledangling(b)
disconnectbond!(b, :B)
@test isdangling(b)
@test !isdoubledangling(b)
disconnectbond!(b, :A)
@test isdangling(b)
@test isdoubledangling(b)
Bonds.relabel!(b, :b)  # TODO Why do I have to specify Bond?
connectbond!(b, :C)
@test isdangling(b)
@test !isdoubledangling(b)
connectbond!(b, :D)
@test !isdangling(b)
@test !isdoubledangling(b)
reconnectbond!(b, :D, :E)
@test !isdangling(b)
@test !isdoubledangling(b)

c = Bond(:b, :C, :E)
@test c == b

@test_throws(ArgumentError, disconnectbond!(c, :F))
@test_throws(ArgumentError, connectbond!(c, :F))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Tests for Nodes

narr = randn(3,4,5,6)
nlabels = [:i, :j, :k ,:l]
n = Node(:N, narr, nlabels)

marr = randn(2,2,7,4,3)
mlabels = [:a, :a, :b, :j, :i]
m = Node(:M, marr, mlabels)

mn = contractnodes(m, n, [:i, :j])
mnarr = tensorcontract(marr, [:a, :c, :b, :j, :i], narr, nlabels)
@test mn.tensor == mnarr

mtrace = contractnodes(m, [:a])
mtracearr = tensortrace(marr, mlabels)
@test mtrace.tensor == mtracearr

mcopy = deepcopy(m)
Nodes.relabel!(mcopy, :K)  # TODO Why do I have to specify Node?
@test mcopy.label === :K
@test m.tensor == mcopy.tensor
@test m.bonds == mcopy.bonds


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Tests for TensorNetworks

a = randn(10,4,5)
b = randn(11,5,3)
c = randn(12,3,4)

abc = ncon((a, b, c), ([-1,1,2], [-2,2,3], [-3,3,1]))
