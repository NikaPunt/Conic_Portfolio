module Distortions

export MINMAXVAR, MINVAR, MAXMINVAR, MAXVAR, Wang

include("MINMAXVAR.jl")
include("MINVAR.jl")
include("MAXMINVAR.jl")
include("MAXVAR.jl")
include("WangTransform.jl")

end