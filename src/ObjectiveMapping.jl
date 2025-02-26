module ObjectiveMapping
using BLUEs
using LinearAlgebra
using Random
using Distributions
using ToeplitzMatrices
using Unitful
using UnitfulLinearAlgebra
# Write your package code here.

export piecewise_linear_regression_matrix
export gauss_markov_mapping, build_mapping_matrices

export meshgrid, bilinear_interpolation_coefficients, construct_correlation_matrix
include("Univariate.jl")
include("Spatial.jl")


end
