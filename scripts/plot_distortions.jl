include("MINVAR.jl")
include("MAXVAR.jl")
include("MINMAXVAR.jl")
include("MAXMINVAR.jl")
include("WangTransform.jl")
using Plots

u_range = 0:0.01:1
u_length = length(u_range)
λs = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8]

#minvar plots
plot(u_range,map(MINVAR,u_range,repeat([λs[1]],u_length)),title="MINVAR",label="λ=0.1",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[2]],u_length)),label="λ=0.2",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[3]],u_length)),label="λ=0.4",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[4]],u_length)),label="λ=0.8",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[5]],u_length)),label="λ=1.6",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[6]],u_length)),label="λ=3.2",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[7]],u_length)),label="λ=6.4",lw=3)
plot!(u_range,map(MINVAR,u_range,repeat([λs[8]],u_length)),label="λ=12.8",lw=3)

#maxvar plots
plot(u_range,map(MAXVAR,u_range,repeat([λs[1]],u_length)),title="MAXVAR",label="λ=0.1",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[2]],u_length)),label="λ=0.2",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[3]],u_length)),label="λ=0.4",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[4]],u_length)),label="λ=0.8",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[5]],u_length)),label="λ=1.6",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[6]],u_length)),label="λ=3.2",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[7]],u_length)),label="λ=6.4",lw=3)
plot!(u_range,map(MAXVAR,u_range,repeat([λs[8]],u_length)),label="λ=12.8",lw=3)

#minmaxvar plots
plot(u_range,map(MINMAXVAR,u_range,repeat([λs[1]],u_length)),title="MINMAXVAR",label="λ=0.1",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[2]],u_length)),label="λ=0.2",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[3]],u_length)),label="λ=0.4",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[4]],u_length)),label="λ=0.8",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[5]],u_length)),label="λ=1.6",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[6]],u_length)),label="λ=3.2",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[7]],u_length)),label="λ=6.4",lw=3)
plot!(u_range,map(MINMAXVAR,u_range,repeat([λs[8]],u_length)),label="λ=12.8",lw=3)


#maxminvar plots
plot(u_range,map(MAXMINVAR,u_range,repeat([λs[1]],u_length)),title="MAXMINVAR",label="λ=0.1",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[2]],u_length)),label="λ=0.2",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[3]],u_length)),label="λ=0.4",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[4]],u_length)),label="λ=0.8",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[5]],u_length)),label="λ=1.6",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[6]],u_length)),label="λ=3.2",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[7]],u_length)),label="λ=6.4",lw=3)
plot!(u_range,map(MAXMINVAR,u_range,repeat([λs[8]],u_length)),label="λ=12.8",lw=3)


#Wang plots
plot(u_range,map(Wang,u_range,repeat([λs[1]],u_length)),title="Wang Transform",label="λ=0.1",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[2]],u_length)),label="λ=0.2",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[3]],u_length)),label="λ=0.4",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[4]],u_length)),label="λ=0.8",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[5]],u_length)),label="λ=1.6",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[6]],u_length)),label="λ=3.2",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[7]],u_length)),label="λ=6.4",lw=3)
plot!(u_range,map(Wang,u_range,repeat([λs[8]],u_length)),label="λ=12.8",lw=3)

