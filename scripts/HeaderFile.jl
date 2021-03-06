# push!(LOAD_PATH,"/home/nikap/Desktop/Masterthesis/Conic_Portfolio/scripts") # On Linux
# push!(LOAD_PATH,"C:\\Users\\nikap\\Desktop\\Conic_Portfolio\\scripts")
push!(LOAD_PATH,pwd()*"/scripts")
using CSV
using DataFrames
using Plots
using Distortions
using Statistics
using LinearAlgebra
using Optim
using JuMP # language
using AmplNLWriter # interface
using Ipopt # solver
using Clp
using PlotlyJS
using Dates
# using GLPK
using Base.Threads: @threads, @spawn
using KNITRO

# In case any of the packages fail to load because they weren't installed, do:
# using Pkg; Pkg.add("<package name>")

include("ICA_assets.jl")
include("implied_moments.jl")
include("getVGParams.jl")
include("VG_Params2MRet.jl")
include("ConicFunctions.jl")
include("CPT_Optimization.jl")
include("getWeights.jl")

