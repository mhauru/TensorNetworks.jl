using TensorNetworks
using TensorNetworks.Bonds
using TensorNetworks.Nodes
using Base.Test
using NCon

a = randn(10,4,5)
b = randn(11,5,3)
c = randn(12,3,4)

abc = ncon((a, b, c), ([-1,1,2], [-2,2,3], [-3,3,1]))

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
