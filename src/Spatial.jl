using Unitful

km = u"km" # for the grid size
cm = u"cm" # for SSH (mapped variable)

meshgrid(x, y) = (x' .* one.(y), one.(x)' .* y)


function find_minimum_positive_distance(distances; min_dist = 1e6km)
    return dhi,ihi = findmin(x -> x > 0km ? x : min_dist,distances)
end

function find_minimum_negative_distance(distances; min_dist = 1e6km)
    return dhi,ihi = findmin(x -> x ≤ 0km ? abs(x) : min_dist,distances)
end

function bilinear_interpolation_coefficients(robs, rx, ry; min_dist = 1e6km)

    Nx, Ny = length(rx), length(ry)
    ydist = [ry[ii] .- robs[2] for ii in eachindex(ry)] # y distance between observation and y grid points
    xdist = [rx[ii] .- robs[1] for ii in eachindex(rx)] # x distance between observation and x grid points

    xhi,ihi = find_minimum_positive_distance(xdist) # find minimum positive distance
    δx,ilo = find_minimum_negative_distance(xdist) # find minimum negative distance
    
    yhi,jhi = find_minimum_positive_distance(ydist) # find minimum positive distance
    δy,jlo = find_minimum_negative_distance(ydist) # find minimum 	
    
    Δx = δx + xhi # grid spacing
    Δy = δy + yhi # grid spacing : y		

    #compute coefficients based on nearest neighbors
    coeffs = zeros(Nx,Ny)
    denom = Δx * Δy
    coeffs[ilo,jlo] = ((Δx - δx)*(Δy - δy))/denom
    coeffs[ilo,jhi] = ((Δx - δx)*(δy))/denom
    coeffs[ihi,jlo] = ((δx)*(Δy - δy))/denom
    coeffs[ihi,jhi] = (δx*δy)/denom

    return coeffs
end

function construct_correlation_matrix(r, Lx, Ly)
    DX = [r[i][1]-r[j][1] for j in eachindex(r), i in eachindex(r)] # doesn't take advantage of symmetry
    DY = [r[i][2]-r[j][2] for j in eachindex(r), i in eachindex(r)] # doesn't take advantage of symmetry
    Rρ = exp.( (-(DX./Lx).^2) .- ((DY./Ly).^2))
    return Rρ
    # Rρ = Rρ + 1e-6I
end