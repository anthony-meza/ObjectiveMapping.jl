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
	using ObjectiveMapping
	using LinearAlgebra, Random
	using Distributions
	using ToeplitzMatrices
	# using Plots, 
	using PlutoUI
	using PythonPlot
	# # pythonplot()
end

# ╔═╡ 908bb398-8442-47c3-8563-cd2f72bd6ab8
begin
	plot_uncertainty(xgrid, x̃, σx̃; color = "red", alpha = 0.2) = plot(xgrid, x̃ - 2σx̃, fillrange = x̃  + 2σx̃, color = color, alzpha = 0.2, label = nothing) 

	plot_uncertainty!(xgrid, x̃, σx̃; color = "red", alpha = 0.2) =  plot!(xgrid, x̃ - 2σx̃, fillrange = x̃  + 2σx̃, color = color, alpha = 0.2, label = nothing) 
	
	plot_uncertainty!(ax, xgrid, x̃, σx̃; color = "red", alpha = 0.2) =  ax.fill_between(xgrid, x̃ - (2*σx̃), x̃ + (2*σx̃), color=color, alpha=alpha, 
		label=nothing)
end

# ╔═╡ 19ca3f75-3005-43fe-9762-b4b1c910677c
begin 
	function parameter_inputs()
		return PlutoUI.combine() do Child
			inputs = [
				md""" **Start of domain (a):** $(
					Child("a", Slider(0:10:300, default=0, show_value=true))
				)""",
				
				md""" **End of domain (b):** $(
					Child("b", Slider(50:10:300, default=100, show_value=true))
				)""",
				
				md""" **Number of points (nx):** $(
					Child("nx", Slider(100:100:2000, default=1000, show_value=true))
				)""",
				
				md""" **Amplitude:** $(
					Child("amplitude", Slider(1:1:20, default=2, show_value=true))
				)""",
				
				md""" **Frequency:** $(
					Child("frequency", Slider(0.01:0.01:0.2, default=0.1, show_value=true))
				)""",
				
				md""" **Mean (μ):** $(
					Child("mean", Slider(0.00:10:100, default=0, show_value=true))
				)""",
				
				md""" **Trend coefficient (β):** $(
					Child("trend", Slider(0.001:0.05:0.5, default=0.0, show_value=true))
				)"""
			]
			
			md"""
			#### Function Parameters
			$(inputs)
			"""
		end
	end
end

# ╔═╡ 8aaae11b-8dc9-4587-9293-2c05703f2ea7
begin 
	function generate_grid(p)
		a = p.a
		b = p.b
		nx = p.nx
		xgrid = collect(LinRange(a, b, nx))
		
		return xgrid
	end
	
	function generate_data(p, xgrid)
		μ = p.mean
		β = p.trend
		amplitude = p.amplitude
		frequency = p.frequency
		
		y_values = (amplitude * sin.(xgrid * frequency)) .+ μ .+ (β .* xgrid)
		
		return y_values
	end
end

# ╔═╡ c0de3bd3-4a78-45c6-9bf7-910f240731a9
@bind parameters parameter_inputs()

# ╔═╡ 1a1bd0e2-feb7-4532-af21-96692b939710
begin # Function to generate the data with the current parameters

	# Get current data
	xgrid = generate_grid(parameters)
	y_analytical = generate_data(parameters, xgrid)

	fig, ax = subplots()
	ax.plot(xgrid, y_analytical, color = "black", label = "True Function", 
      alpha = 1, linestyle = "--", lw = 2)
	fig
end

# ╔═╡ d8db1a46-f40e-4f86-a61b-21cf319ceceb
function observation_inputs()
    return PlutoUI.combine() do Child
        
        inputs = [
            md""" **Noise standard deviation (σn):** $(
                Child("σn", Slider(0.0:0.1:10.0, default=5.0, show_value=true))
            )""",
            
            md""" **Observation percentage (%):** $(
                Child("obs_percent", Slider(1:1:50, default=10, show_value=true))
            )"""
        ]
        
        md"""
        #### Observation Parameters
        $(inputs)
        """
		
    end
end

# ╔═╡ 214be50e-c796-4861-a3b1-85f6ea5d5b7c
@bind observation_params observation_inputs()

# ╔═╡ 21f79e54-396f-4f8d-88bd-1478372185f0
begin 
	nx = parameters.nx
	σn = observation_params.σn
	nobs =Int(round(observation_params.obs_percent / 100 * nx)) #observe some of the function
	obsgrid = rand(Uniform(minimum(xgrid), maximum(xgrid)), nobs) 
	yobs = generate_data(parameters, obsgrid) .+ rand(Normal(0, σn), nobs)

	fig1, ax1 = subplots()
	ax1.scatter(obsgrid, yobs, label = "Contaminated Observations", 
        alpha = 0.1, color = "black")
	ax1.plot(xgrid, y_analytical, color = "black", label = "True Function", 
      alpha = 1, linestyle = "--", lw = 2)
	ax1.set_xlabel("time")
	fig1
end

# ╔═╡ 87ee8498-c807-4364-bc65-c504a13aa10c
begin 
	H = ObjectiveMapping.piecewise_linear_regression_matrix(xgrid, obsgrid)
	yobs_gridded = pinv(H) * obsgrid
	Lt = 19 #decorrelation time scale
	σx = 5 #prior variance
	
	x̃, σx̃ = ObjectiveMapping.gauss_markov_mapping(ObjectiveMapping.StandardMapping(), yobs, obsgrid, xgrid; Lt = Lt, σx = σx, σn = σn)
	
	x̃b, σx̃b = ObjectiveMapping.gauss_markov_mapping(ObjectiveMapping.BLUESMapping(), yobs, obsgrid, xgrid; Lt = Lt, σx = σx, σn = σn)
		
	# First create a base plot with contaminated observations and true function
	fig2, axs = subplots(1, 2, figsize = (10, 5))
	for axx in axs
		axx.scatter(obsgrid, yobs, label = "Contaminated Observations", 
		        alpha = 0.1, c = "black")
		axx.plot(xgrid, y_analytical, color = "black", label = "True Function", 
		      alpha = 1, linestyle = "--", lw = 2)
	end
	axs[0].plot(xgrid, x̃, color = "red", label = "Gauss-Markov")
	plot_uncertainty!(axs[0], xgrid, x̃, σx̃; color = "orange", alpha = 0.3)
	
	axs[1].plot(xgrid, x̃b, color = "purple", label = "BLUES.jl")
	plot_uncertainty!(axs[1], xgrid, x̃b, σx̃b; color = "purple", alpha = 0.3)
	
	fig2

end

# ╔═╡ Cell order:
# ╠═44d6fbeb-0253-4a5b-b810-3108a88963c3
# ╠═e3b69a7f-cad0-4535-b691-a2d53c586581
# ╠═908bb398-8442-47c3-8563-cd2f72bd6ab8
# ╠═19ca3f75-3005-43fe-9762-b4b1c910677c
# ╠═8aaae11b-8dc9-4587-9293-2c05703f2ea7
# ╠═c0de3bd3-4a78-45c6-9bf7-910f240731a9
# ╠═1a1bd0e2-feb7-4532-af21-96692b939710
# ╠═d8db1a46-f40e-4f86-a61b-21cf319ceceb
# ╠═214be50e-c796-4861-a3b1-85f6ea5d5b7c
# ╠═21f79e54-396f-4f8d-88bd-1478372185f0
# ╠═87ee8498-c807-4364-bc65-c504a13aa10c
