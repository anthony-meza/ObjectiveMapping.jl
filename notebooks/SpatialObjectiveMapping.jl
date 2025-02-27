### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 44d6fbeb-0253-4a5b-b810-3108a88963c3
begin
	import Pkg
	Pkg.activate("../")
	Pkg.instantiate()
end

# ╔═╡ e3b69a7f-cad0-4535-b691-a2d53c586581
begin
	using BLUEs
	using ObjectiveMapping
	using LinearAlgebra, Random
	using Distributions
	using ToeplitzMatrices
	using SparseArrays
	using Plots
	using PlutoUI
	# using PythonPlot
	using Statistics, Distributions
	using Unitful, UnitfulLinearAlgebra
	using InteractiveUtils
	using Random
	# pythonplot()
end

# ╔═╡ a48c4115-8e05-4510-b215-94bd9a374941
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

# ╔═╡ b9845c85-06f6-4950-abd2-69f516c51e1b
begin 
    function rosenbrock_parameters()
        return PlutoUI.combine() do Child
            inputs = [
                md""" **a:** $(
                    Child("a", Slider(1:0.1:4, default=3, show_value=true))
                ) """,
                
                md""" **b:** $(
                    Child("b", Slider(0.1:0.1:5.0, default=0.5, show_value=true))
                )""",
            ]
            
            md"""
            #### Rosenbrock Function Parameters: $f(x,y)=(a-x)^{2}+b(y-x^{2})^{2}$
            $(inputs)
            """
        end
    end
	# In another cell, bind the parameters
	@bind rosen_params rosenbrock_parameters()
end

# ╔═╡ 4b99b673-d890-4aa2-acc3-fb545b4fc479
begin 
	#store the 2D ordered points
	gx, gy = meshgrid(rx,ry)

	rosenbrok(x, y) = (rosen_params.a- x)^2 + rosen_params.b * (y - x^2)^2
	ηtrue = 1cm .* rosenbrok.(gx ./ mean(gx), gy ./ mean(gy))
	vmin, vmax = extrema(ustrip.(ηtrue))
	vmin, vmax = (vmin * 0.9, vmax * 1.1)
	levels = LinRange(vmin, vmax, 20)
	cmap = :Spectral

	contourf(rx, ry, ηtrue,xlabel="zonal distance",
	    ylabel="meridional distance",clabels=true,title="true SSH",
		fill=true, levels = levels, clims=(vmin, vmax), cmap = cmap)

end

# ╔═╡ 2c732a10-c15c-4b19-a16d-9ea5750ea441
begin 
    function observation_parameters()
        return PlutoUI.combine() do Child
            inputs = [
                md""" **Observation Density:** $(
                    Child("obs_percent", Slider(1:1:20, default=5, show_value=true))
                ) %""",
                
                md""" **Observation Noise (σₙ):** $(
                    Child("noise", Slider(0.1:0.1:2.0, default=0.5, show_value=true))
                ) cm""",
            ]
            
            md"""
            #### Observation Parameters
            $(inputs)
            """
        end
    end
end

# ╔═╡ 80367e0f-47fe-4180-be46-eda106e0cf15
# In another cell, bind the parameters
@bind obs_settings observation_parameters()

# ╔═╡ 69deeb60-46a6-470a-8e09-47dfc0cd8bce
begin 
	obs_percent = obs_settings.obs_percent / 100  # Convert percentage to fraction
	Nobs = Int(round(obs_percent .* length(r))) # number of observations
	robs = [(ΔX*rand(),ΔY*rand()) for i in 1:Nobs ] # uniformly sampled
	
	# make E matrix for these observations, use bilinear interpolation
	# get bilinear interpolation coefficients
	E = zeros(Nobs,Nx*Ny)
	for oo in eachindex(robs)
	    coeffs = bilinear_interpolation_coefficients(robs[oo], rx, ry)
	    E[oo,:] = vec(coeffs)
	end
	
	# how much observational noise
	σₙ = obs_settings.noise * cm
	# get the noise covariance
	Cnn = Diagonal(fill(σₙ^2,Nobs))
	# Sample the true field
	yvals = E*vec(ηtrue') + (1cm * rand(Normal(0, ustrip(σₙ)), Nobs))

	rxobs = [robs[oo][1] for oo in eachindex(robs)]
	ryobs = [robs[oo][2] for oo in eachindex(robs)]
	
	p = contourf(rx, ry, ηtrue,xlabel="zonal distance",
	    ylabel="meridional distance",clabels=true,
	    title="true SSH with $Nobs obs",fill=true, 
		levels = levels, clims=(vmin, vmax), cmap = cmap)
	scatter!(p, rxobs,ryobs,zcolor=yvals,label="y",
		cbar=true,markersize=6, 
	levels = levels, clims=(vmin, vmax), cmap = cmap)

end

# ╔═╡ 8ae938a0-de85-47ab-89c8-436d2ec8d2d1
begin 
	function decorrelation_inputs()
        return PlutoUI.combine() do Child
            inputs = [
                md""" **Zonal Decorrelation Length Scale (Lx):** $(
                    Child("Lx", Slider(50:10:1000, default=500, show_value=true))
                ) km""",
                
                md""" **Meridional Decorrelation Length Scale (Ly):** $(
                    Child("Ly", Slider(100:25:500, default=500, show_value=true))
                ) km""",
            ]
            
            md"""
            #### Domain Size Parameters
            $(inputs)
            """
        end
    end
end

# ╔═╡ 1e86504d-d29c-471b-90e6-402dd4e3e4d3
@bind decorr_scales decorrelation_inputs()

# ╔═╡ bb27f5f9-54a5-41c0-958b-160488c6f552
begin 
	# set decorrelation lengthscales
	Lx = decorr_scales.Lx * km; Ly = decorr_scales.Ly * km;
	Rρ = construct_correlation_matrix(r, Lx, Ly)
	
	p0 = heatmap(Rρ, title = "empirical correlation matrix") # not simply monotonic
	p1 = scatter(rxobs, yvals, xlabel = "zonal distance", ylabel = "SSH observation", label = nothing)
	p2 = scatter(ryobs, yvals, xlabel = "meridional distance", ylabel = "SSH observation", label = nothing)
	l = @layout [
    a{0.5w} [b; c]]
	plot(p0, p1, p2, size = (800, 500),  layout = l)
end

# ╔═╡ dc3bb5fe-d692-47dd-ab6a-1a9576db604c


# ╔═╡ 4aef1f8b-ac28-4e25-a319-3a94052c4cf4
begin 
	σ² = std(yvals)^2
	Cxx = σ² .* Rρ
	y = Estimate(yvals,Cnn)
	
	nx̃ = length(r)
	x0vals = zeros(eltype(ηtrue[1]),nx̃) .+ mean(y.v) # first guess
	x0 = Estimate(x0vals, Cxx)
	
	x̃ = combine(x0, y, E)
	x̃field = reshape(x̃.v,Nx,Ny); # turn it back into 2D

	# Create the first contour plot with horizontal colorbar
	p4 = contourf(rx, ry, x̃field', 
	    clims=(vmin, vmax), levels=levels, 
	    clabels=true, title="estimated SSH",
	    ylabel="meridional distance", xlabel="zonal distance",
	    colorbar=true,
	    colorbar_horizontal=true,
	    bottom_margin=20Plots.mm, cmap = cmap)  # Add margin for the colorbar
	
	# Create the second contour plot with horizontal colorbar
	p5 = contourf(rx, ry, ηtrue, 
	    clims=(vmin, vmax), levels=levels, 
	    clabels=true, title="true SSH",
	    xlabel="zonal distance", 
	    colorbar=true,
	    colorbar_horizontal=true,
	    bottom_margin=20Plots.mm, cmap = cmap)  # Add margin for the colorbar
	
	# Combine plots side by side
	plot(p4, p5, layout=(1,2), size=(900, 450))
end

# ╔═╡ Cell order:
# ╟─44d6fbeb-0253-4a5b-b810-3108a88963c3
# ╠═e3b69a7f-cad0-4535-b691-a2d53c586581
# ╠═a48c4115-8e05-4510-b215-94bd9a374941
# ╟─b9845c85-06f6-4950-abd2-69f516c51e1b
# ╠═4b99b673-d890-4aa2-acc3-fb545b4fc479
# ╟─2c732a10-c15c-4b19-a16d-9ea5750ea441
# ╟─80367e0f-47fe-4180-be46-eda106e0cf15
# ╠═69deeb60-46a6-470a-8e09-47dfc0cd8bce
# ╟─8ae938a0-de85-47ab-89c8-436d2ec8d2d1
# ╠═1e86504d-d29c-471b-90e6-402dd4e3e4d3
# ╠═bb27f5f9-54a5-41c0-958b-160488c6f552
# ╟─dc3bb5fe-d692-47dd-ab6a-1a9576db604c
# ╠═4aef1f8b-ac28-4e25-a319-3a94052c4cf4
