using TensorNetworks
using TensorNetworks.Bonds
using TensorNetworks.Nodes
using Base.Test
using NCon

a = randn(10,4,5)
b = randn(11,5,3)
c = randn(12,3,4)

abc = ncon((a, b, c), ([-1,1,2], [-2,2,3], [-3,3,1]))

