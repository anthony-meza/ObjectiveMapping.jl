module ObjectiveMapping
using BLUEs
using LinearAlgebra
using Random
using Distributions
using ToeplitzMatrices
# Write your package code here.

"""
    piecewise_linear_regression_matrix(xgrid, ygrid)

Constructs a piecewise linear interpolation matrix that maps between two grids.

# Arguments
- `xgrid::Vector`: Source grid points
- `ygrid::Vector`: Target grid points

# Returns
- `Matrix`: Interpolation matrix H where H[i,j] contains the weights for linear interpolation
"""
function piecewise_linear_regression_matrix(xgrid, ygrid)
    nx = length(xgrid)
    ny = length(ygrid)
    H = zeros(ny, nx)
    
    for (i, yg) in enumerate(ygrid)
        # Find closest points in xgrid for interpolation
        δxlo, ixlo = findmin(x -> (x > 0) ? abs(x) : Inf, yg .- xgrid)   # Left point
        δxhigh, ixhigh = findmin(x -> (x ≤ 0) ? abs(x) : Inf, yg .- xgrid) # Right point
        
        # Compute interpolation weights
        if ixhigh != ixlo
            Δx = xgrid[ixhigh] - xgrid[ixlo]
            a = (yg - xgrid[ixlo]) / Δx
            H[i, ixlo] = 1 - a
            H[i, ixhigh] = a
        else 
            # Handle edge case where points coincide
            H[i, ixlo] = 1.0
        end
    end
    
    # Replace any NaN values with zeros for numerical stability
    H[isnan.(H)] .= 0.0
    return H
end

"""
    build_mapping_matrices(xgrid, ygrid; Lt=5, σx=3, σn=0.5)

Helper function to construct the interpolation matrix and covariance matrices needed for 
Gauss-Markov mapping.

# Returns
- `Tuple`: (H, Cxx, Cnn) where:
    - H: Interpolation matrix mapping between grids
    - Cxx: Background error covariance matrix
    - Cnn: Observation error covariance matrix
"""
function build_mapping_matrices(xgrid, ygrid; Lt=5, σx=3, σn=0.5)
    # Compute interpolation matrix
    H = piecewise_linear_regression_matrix(xgrid, ygrid)
    
    # Construct background error covariance matrix
    DX = [j - i for i in xgrid, j in xgrid]  # Distance matrix
    Cxx = σx .* exp.(-(DX ./ Lt).^2)         # Background error covariance
    
    # Construct observation error covariance matrix
    Cnn = σn * I(length(ygrid))              # Observation error covariance
    
    return H, Cxx, Cnn
end

# Abstract type for different mapping methods
abstract type MappingMethod end

# Concrete types for specific mapping methods
struct StandardMapping <: MappingMethod end
struct BLUESMapping <: MappingMethod end

"""
    gauss_markov_mapping([method::MappingMethod], yobs, ygrid, xgrid; kwargs...)

Unified interface for Gauss-Markov mapping with different implementation methods.

# Arguments
- `method::MappingMethod`: Mapping method (StandardMapping() or BLUESMapping())
- `yobs::Vector`: Observed values
- `ygrid::Vector`: Grid points of observations
- `xgrid::Vector`: Grid points for interpolation
- `Lt::Real`: Length scale for correlation (default: 5)
- `σx::Real`: Prior variance (default: 3)
- `σn::Real`: Observation noise variance (default: 0.5)
- `x0::Union{Vector,Nothing}`: Optional background state (default: nothing)

# Returns
- Tuple(Vector, Vector): (Interpolated values, Uncertainty estimates)
"""
function gauss_markov_mapping(::StandardMapping, yobs, ygrid, xgrid; 
                            Lt=5, σx=3, σn=0.5, x0=nothing)
    H, Cxx, Cnn = build_mapping_matrices(xgrid, ygrid; Lt=Lt, σx=σx, σn=σn)
    
    # Compute posterior covariance
    P = Cxx - Cxx * H' * inv(H * Cxx * H' + Cnn) * H * Cxx
    
    # Compute optimal interpolation
    if isnothing(x0)
        x̃ = Cxx * H' * inv(H*Cxx*H' + Cnn) * yobs
    else
        x̃ = x0 + Cxx * H' * inv(H*Cxx*H' + Cnn) * (yobs .- H*x0)
    end
    
    return x̃, sqrt.(diag(P))
end

function gauss_markov_mapping(::BLUESMapping, yobs, ygrid, xgrid; 
                            Lt=5, σx=3, σn=0.5, x0=nothing)
    H, Cxx, Cnn = build_mapping_matrices(xgrid, ygrid; Lt=Lt, σx=σx, σn=σn)
    
    # Create estimates with uncertainties
    yp = Estimate(yobs, Cnn)
    #if no first guess provided, equivalent to zero first guess 
    x0p = Estimate(isnothing(x0) ? zeros(length(xgrid)) : x0, Cxx) 
    
    # Combine estimates using BLUES
    x̃ = combine(x0p, yp, H)
    return (x̃.v, x̃.σ)
end

end
