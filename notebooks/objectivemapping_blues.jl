### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 0e92b6d5-20c9-411c-959b-4e19639fb574
begin
	ENV["UNITFUL_FANCY_EXPONENTS"] = true
	import Pkg; Pkg.activate(".")
	Pkg.instantiate()
	using BLUEs
	using Test
	using LinearAlgebra
	using Statistics
	using Unitful
	using UnitfulLinearAlgebra
	using Measurements
	using ToeplitzMatrices
	using SparseArrays
	using Plots
	using Random
	using InteractiveUtils
	using PlutoUI
	using Pluto
	plotlyjs()
	#gr()
end


# ╔═╡ a72352ab-cdfb-42db-85de-ea88132ba9dd
md""" # Objective mapping with `BLUEs.jl`"""

# ╔═╡ 7d13a231-8ee5-459b-9c0e-dd81ab87bda1
md""" ## 2D objective map of sea surface height """

# ╔═╡ e3b69a7f-cad0-4535-b691-a2d53c586581
begin 
	km = u"km" # for the grid size
	cm = u"cm" # for SSH (mapped variable)
	Nx = 50 # number of gridpoint in first (zonal) direction
	ΔX = 1000km # domain size in zonal direction
	Ny = 40 # number of gridpoints in second (meridional) direction
	ΔY = 500km

	
	rx = range(0km,ΔX,length=Nx) # make grid axis number 1: zonal distance
	ry = range(0km,ΔY,length=Ny)  # grid axis number 2: meridional distance

	# turn the 2D grid into a 1D bookkeeping system
	# a vector that gives x location as first element, y location as second element
	r = [(rx[i],ry[j])  for j in eachindex(ry) for i in eachindex(rx)] 
end

# ╔═╡ 4854bf8b-21d4-42a3-bfe9-3cc77d790544
# location of first gridpoint
r[1]

# ╔═╡ e42ecaa5-052d-47e3-aaea-9905f6c4b508
# location of second gridpoint
r[2]

# ╔═╡ 96ad4635-70a1-4e41-912e-28cca0b33e75
#location of 51st gridpoint
r[51]

# ╔═╡ 16a42d32-853b-417c-845b-98da700e1be0
begin
	# set decorrelation lengthscales
	Lx = 300km; Ly = 100km;
	Rρ = [exp( -((r[i][1]-r[j][1])/Lx)^2 - ((r[i][2] - r[j][2])/Ly)^2) for j in eachindex(r), i in eachindex(r)] # doesn't take advantage of symmetry
end

# ╔═╡ 8cd47f91-c3b5-4522-8ac3-323e22601633
# show a slice

# ╔═╡ 412e835f-d518-45c1-b6f0-38e58a3d8bfc
plot(Rρ[20,:],xlabel="grid point (1D index)",ylabel="ρ", seriestype=:scatter) # not simply monotonic

# ╔═╡ 4dc7e663-f34e-4cf9-91c4-def74172ef57
# Here I need to produce some "synthetic" data
# these are the extra steps
begin 
	Rρ_posdef = Rρ + 1e-6I
	
	Rρ12 = cholesky(Rρ_posdef) # cholesky 
end

# ╔═╡ 38b8e924-6ec1-41f0-a972-756ce1c5cd4d
begin
	# turn correlation matrix into autocovariance matrix: requires variance info
	σ² = (50cm)^2

	#construct synthetic data vector 
	xtrue = √σ²*Rρ12.L*randn(Nx*Ny) 

	#reshape into a 2D field
	xtruefield = reshape(xtrue,Nx,Ny)
end

# ╔═╡ 3fb8271b-bcbd-4a38-9589-8723c490fbba
p1 = contourf(rx,ry,ustrip.(xtruefield'),xlabel="zonal distance",ylabel="meridional distance",clabels=true,cbar=false,title="true SSH")

# ╔═╡ 6510d727-5612-4605-9f35-9ce8eda98f02


# ╔═╡ bdafc085-e15e-4130-bca1-7abf5e724025
begin 
	#determine random locations for synthetic observations
	Nobs = 20 # number of observations
	robs = [(ΔX*rand(),ΔY*rand()) for i in 1:Nobs ] # uniformly sampled

end

# ╔═╡ 37c5091f-3cf5-46b0-8874-59a047a191e6
begin 
	function find_minimum_positive_distance(distances; min_dist = 1e6km)
		return dhi,ihi = findmin(x -> x > 0km ? x : min_dist,distances)
	end
	
	function find_minimum_negative_distance(distances; min_dist = 1e6km)
		return dhi,ihi = findmin(x -> x ≤ 0km ? abs(x) : min_dist,distances)
	end
	
	function bilinear_interpolation_coefficients(robs, rx, ry; min_dist = 1e6km)
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
end

# ╔═╡ 64f8cb6c-9c8a-4f08-b970-697708da228b
begin 
	# make E matrix for these observations, use bilinear interpolation
	# get bilinear interpolation coefficients
	E = zeros(Nobs,Nx*Ny)
	for oo in eachindex(robs)
		coeffs = bilinear_interpolation_coefficients(robs[oo], rx, ry)
		E[oo,:] = vec(coeffs)
	end
	# check that each row sums to one
	sum(E,dims=2)
end

# ╔═╡ 180bce23-cc3c-4869-ae30-2cd0d61aa34f
begin 
	# how much observational noise
	σₙ = 1cm
	# get the noise covariance
	Cnn = Diagonal(fill(σₙ^2,Nobs))
	# Sample the true field
	yvals = E*xtrue + σₙ*randn(Nobs) 

	y = Estimate(yvals,Cnn)
end

# ╔═╡ 007e1141-4a33-4538-a089-b3238117af3d
begin
	# add the observations to the plot
	rxobs = [robs[oo][1] for oo in eachindex(robs)]

	ryobs = [robs[oo][2] for oo in eachindex(robs)]
	
	p2 = contour(rx,ry,ustrip.(xtruefield'),xlabel="zonal distance",ylabel="meridional distance",clabels=true,title="true SSH with $Nobs obs",fill=true)
	scatter!(p2, rxobs,ryobs,zcolor=ustrip.(y.v),label="y",cbar=false,markersize=6) 
end

# ╔═╡ a8832dcd-ce4f-4967-8ed7-5c24d05c7a95
## Now we have synthetic observations
# Let's see if the true solution can be backed out from the sparse obs.

begin 
	nxtrue = length(ustrip.(xtrue))
	#use `BLUEs.jl` to package as an `Estimate`
	x0vals = zeros(eltype(xtrue[1]),nxtrue) # first guess

	Cxx = σ²*Rρ_posdef

	#establish first guess
	x0 = Estimate(x0vals, Cxx)

	#estimate values
	x̃ = combine(x0, y, E)

	x̃field = reshape(x̃.v,Nx,Ny) # turn it back into 2D
end

# ╔═╡ 371c28b5-90e0-4f69-ac9b-e706fe960d07
begin
	p3 = contourf(rx,ry,x̃field',xlabel="zonal distance",ylabel="meridional distance",title="SSH objective map")
	scatter!(p3, rxobs,ryobs,zcolor=ustrip.(y.v),label="y",ms=6,cbar=false) 
end

# ╔═╡ 573c07ba-2c7a-4b95-9ed6-dd0563c257e5
begin
	σₓ = x̃.σ
	σx̃field = reshape(x̃.σ,Nx,Ny) # turn it back into 2D
	contourf(rx,ry,σx̃field',xlabel="zonal distance",
							ylabel="meridional distance",
							title="SSH uncertainty, " *  string(unit(σx̃field[1])),
							clabels=true,
							cbar=false)
	scatter!(rxobs,ryobs,color=:white,label="y",ms=6) 
end

# ╔═╡ 890ab7e1-2b65-4454-8ca4-e2695cde3f35
begin
	l = @layout [b c]
	plot(p2, p3, layout = l, size = (1000, 500))
end

# ╔═╡ 545a3bc2-f399-41eb-99fc-1ea40239d856
md""" ## how well is the data fit? """

# ╔═╡ 387f5555-c3f3-4b80-849d-8d65491fc6f8
plot(y.v,(E*x̃).v,xlabel="y",ylabel="ỹ", seriestype=:scatter)

# ╔═╡ 935faa9d-990d-4bfa-b70c-b11bc824e20b
plot(y.v,y.v-(E*x̃).v,xlabel="y",ylabel="ñ", seriestype=:scatter)

# ╔═╡ Cell order:
# ╠═0e92b6d5-20c9-411c-959b-4e19639fb574
# ╠═a72352ab-cdfb-42db-85de-ea88132ba9dd
# ╟─7d13a231-8ee5-459b-9c0e-dd81ab87bda1
# ╠═e3b69a7f-cad0-4535-b691-a2d53c586581
# ╠═4854bf8b-21d4-42a3-bfe9-3cc77d790544
# ╠═e42ecaa5-052d-47e3-aaea-9905f6c4b508
# ╠═96ad4635-70a1-4e41-912e-28cca0b33e75
# ╠═16a42d32-853b-417c-845b-98da700e1be0
# ╠═8cd47f91-c3b5-4522-8ac3-323e22601633
# ╠═412e835f-d518-45c1-b6f0-38e58a3d8bfc
# ╠═4dc7e663-f34e-4cf9-91c4-def74172ef57
# ╠═38b8e924-6ec1-41f0-a972-756ce1c5cd4d
# ╠═3fb8271b-bcbd-4a38-9589-8723c490fbba
# ╠═6510d727-5612-4605-9f35-9ce8eda98f02
# ╠═bdafc085-e15e-4130-bca1-7abf5e724025
# ╠═37c5091f-3cf5-46b0-8874-59a047a191e6
# ╠═64f8cb6c-9c8a-4f08-b970-697708da228b
# ╠═180bce23-cc3c-4869-ae30-2cd0d61aa34f
# ╠═007e1141-4a33-4538-a089-b3238117af3d
# ╠═a8832dcd-ce4f-4967-8ed7-5c24d05c7a95
# ╠═371c28b5-90e0-4f69-ac9b-e706fe960d07
# ╠═573c07ba-2c7a-4b95-9ed6-dd0563c257e5
# ╠═890ab7e1-2b65-4454-8ca4-e2695cde3f35
# ╠═545a3bc2-f399-41eb-99fc-1ea40239d856
# ╠═387f5555-c3f3-4b80-849d-8d65491fc6f8
# ╠═935faa9d-990d-4bfa-b70c-b11bc824e20b
