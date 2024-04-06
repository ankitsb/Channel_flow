# %%
using Random
using Printf
using JLD2
using CUDA
using CSV
using DataFrames
using Dierckx
using SeawaterPolynomials.TEOS10
using GibbsSeaWater
using Statistics

using NetCDF
using Glob
using Interpolations
using Adapt
using Roots

using Oceananigans
using Oceananigans.Units: minute, minutes, hour, hours, day, second, meters
using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation, implicit_diffusion_solver
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.BuoyancyModels: ρ′, thermal_expansionᶜᶜᶜ, haline_contractionᶜᶜᶜ, ∂z_b
using Oceananigans.Fields
using Oceananigans.Operators
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑxᶠᵃᵃ,ℑyᵃᶜᵃ, ℑyᵃᶠᵃ,ℑzᵃᵃᶜ, ℑzᵃᵃᶠ,ℑxyᶜᶜᵃ,ℑxzᶜᵃᶜ,ℑyzᵃᶜᶜ
using Oceananigans.Operators: δxᶜᵃᵃ,δxᶠᵃᵃ,δyᵃᶜᵃ,δyᵃᶠᵃ,δzᵃᵃᶜ,δzᵃᵃᶠ
using Oceananigans.Models: seawater_density

# using Oceanostics: IsotropicKineticEnergyDissipationRate, ZShearProductionRate

δ = 1
Lx = 4π*δ
Ly = 2π*δ
Lz = 2*δ
Nx = 180
Ny = 120
Nz = 192

Re_t = 180
ub = 2/3

z_faces(k) = - Lz/2 * cos(π * (k - 1) / Nz)
# z_faces(k) = Lz/2 * (k - Nz/2) / (Nz/2)

# Nz = 132
# z_faces(k) = Lz/2 * tanh((y - 96) / 90) / 0.788

# Nz = 198
# z_faces(k) = tanh((k - 100) / 40) / 0.9859328

Nh = 3

grid = RectilinearGrid(GPU(),
        size = (Nx, Ny, Nz),
        topology=(Oceananigans.Periodic, Oceananigans.Periodic, Bounded),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces,
        halo = (Nh, Nh, Nh),
)

@info "Build a grid:"
@show grid

# %%
# boundary conditions: inflow and outflow in y
# u_bc = OpenBoundaryCondition(1.0)
u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0), bottom = ValueBoundaryCondition(0))
v_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0), bottom = ValueBoundaryCondition(0))
w_bcs = FieldBoundaryConditions(top = OpenBoundaryCondition(0) , bottom = OpenBoundaryCondition(0) )

dp_dx(x, y, z, t) = 0.00176
u_forcing = Forcing(dp_dx)

@info "Conditions required for model are done"
# %%
model = NonhydrostaticModel(; grid, 
                        coriolis = nothing,
                        buoyancy = nothing,
                        closure = ScalarDiffusivity(ν=2.3310e-4), #, κ=1/395 #VerticallyImplicitTimeDiscretization(), 
                        timestepper = :RungeKutta3,
                        advection = WENO(grid, order=5), #UpwindBiasedFifthOrder(),
                        # tracers = :mass,
                        boundary_conditions = (u=u_bcs, v=v_bcs, w=w_bcs),
                        forcing = (u=u_forcing, ),
)

@info "Constructed a model"
@show model

# %%
# noise(x, y, z) = 1e-2 * randn() * (-(z)^2 + 1)
# U∞(x, y, z) = 60exp(-z^20)-22.1 + noise(x, y, z)
U∞(x, y, z) = (1 + randn()/3) * ub * z * exp(-9/2 * z^2) * cos(π * y * Re_t / 100*δ) + 3 * ub * ((z+1) - 0.5 * (z+1)^2)
V∞(x, y, z) = (1 + randn()/3) * ub * z * exp(-9/2 * z^2) * sin(2π * x * Re_t / 250*δ)
# U∞(x, y, z) = (ub+0.1)*(exp(-z^20) - 0.36787944) / 0.632103306 # + noise(x, y, z)
# vᵢ(x, y, z) = noise(x, y, z)
mᵢ(x, y, z) = 1.0
set!(model, u=U∞, v=V∞, w=V∞) #, mass=mᵢ


wall_clock = time_ns()

function progress(sim)
        u_max = maximum(abs, interior(model.velocities.u))
        v_max = maximum(abs, interior(model.velocities.v))
        w_max = maximum(abs, interior(model.velocities.w))
        @info(@sprintf("Iter: %d, time: %s, Δt: %s, max|u|: %.2f, max|v|: %.5f, max|w|: %.5f, wall time: %s\n",
                sim.model.clock.iteration,
                prettytime(sim.model.clock.time),
                prettytime(sim.Δt),
                u_max, v_max, w_max,
                prettytime(1e-9 * (time_ns() - wall_clock[1])), ))

        return nothing
end

simulation = Simulation(model, Δt=1e-4, stop_time=1hour)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(200))

wizard = TimeStepWizard(cfl=0.2, max_change=1.05, max_Δt=1, min_Δt=1e-5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

@info "model initial conditions are Set!!"

# Output: primitive fields + computations
# u, v, w, pHY′, pNHS, mass  = merge(model.velocities, model.pressures, model.tracers)
# p = Field(model.pressures.pNHS + model.pressures.pHY′)
outputs = model.velocities #merge(model.velocities, model.tracers)

simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs;
        filename = joinpath(@__DIR__, "channel_flow-new.nc"),
        schedule = TimeInterval(6),
        overwrite_existing = true,
        with_halos = true
)

# #####
# ##### Build checkpointer and output writer
# #####
simulation.output_writers[:checkpointer] = Checkpointer(model,
                            schedule = TimeInterval(30),
                            prefix = "checkpoint",
                            cleanup=true)
@info "Output files gererated"


run!(simulation) #, pickup=true)