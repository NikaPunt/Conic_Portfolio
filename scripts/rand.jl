# using Pkg
# Pkg.add("GLM")
using GLM
include("HeaderFile.jl")
using PlotlyJS

begin
    x = vcat(collect(0:0.001:0.02),collect(0.02:0.01:1)) ;
    one =   PlotlyJS.scatter(;x=x,  y=MAXMINVAR.(x,a[1]), mode="lines",line_width=5)
    two =   PlotlyJS.scatter(;x=x,  y=MAXMINVAR.(x,a[2]), mode="lines",line_width=5)
    three = PlotlyJS.scatter(;x=x,  y=MAXMINVAR.(x,a[3]), mode="lines",line_width=5)
    four =  PlotlyJS.scatter(;x=x,  y=MAXMINVAR.(x,a[4]), mode="lines",line_width=5)
    five =  PlotlyJS.scatter(;x=x,  y=MAXMINVAR.(x,a[5]), mode="lines",line_width=5)    
    six =   PlotlyJS.scatter(;x=x,  y=MAXMINVAR.(x,a[6]), mode="lines",line_width=5)
    # trace2 = PlotlyJS.scatter(;x=[0.09,0.16,0.24,0.3,0.34,0.37], y=[0.92365,0.8089,0.65,0.513,0.4252,0.3706],
    #                   mode="markers+text",
    #                   textposition="top left",
    #                   text= ["$(a[6])", "$(a[5])", "$(a[4])", "$(a[3])", "$(a[2])","$(a[1])"],
    #                   marker_size=5, textfont_family="Times New Roman",marker_color="rgb(0,0,0)")
    p = PlotlyJS.plot([one,two,three,four,five,six],PlotlyJS.Layout(;xaxis_showgrid=false,
        showlegend=false, 
        yaxis_showgrid=false,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"))
    PlotlyJS.savefig(p, "plots/beamerfront.pdf")
end


a = vec( five_γ_short[1]'*Rtrns)
a2 = vec(five_γ_short[2]'*Rtrns)
a3 = vec(five_γ_short[3]'*Rtrns)
a4 = vec(five_γ_short[4]'*Rtrns)
a5 = vec(five_γ_short[5]'*Rtrns)
b = vec(w_optimMPT_short'*Rtrns)
c = vec(w_optimcvar99_short'*Rtrns)
data = DataFrame(a = a,a2 = a2,a3=a3,a4=a4,a5=a5, b = b, c = c)

fm = @formula(a ~ b + c)
fm2 = @formula(a2 ~ b + c)
fm3 = @formula(a3 ~ b + c)
fm4 = @formula(a4 ~ b + c)
fm5 = @formula(a5 ~ b + c)
linearRegressor = lm(fm, data)
linearRegressor2 = lm(fm2, data)
linearRegressor3 = lm(fm3, data)
linearRegressor4 = lm(fm4, data)
linearRegressor5 = lm(fm5, data)

M=10000
one = [MAXMINVAR(m/M,12) for m=0:M]
plot(range(0,stop=1,length=M+1),one)
two = [one[m]-one[m-1] for m=2:(M+1)]

plot(range(0,stop=1,length=M),two)

function vec_triu_loop(M::AbstractMatrix{T}) where T
    m, n = size(M)
    m == n || throw(error("not square"))
    l = n*(n+1) ÷ 2
    v = Vector{T}(undef,l)
    k = 0
    @inbounds for i in 1:n
        for j in 1:i
            v[k + j] = M[j, i]
        end
        k += i
    end
    v
end

vec_triu_loop(c.U)


a = vec( five_γ[1]'*Rtrns)
a2 = vec(five_γ[2]'*Rtrns)
a3 = vec(five_γ[3]'*Rtrns)
a4 = vec(five_γ[4]'*Rtrns)
a5 = vec(five_γ[5]'*Rtrns)
b = vec(w_optimMPT'*Rtrns)
c = vec(w_optimcvar'*Rtrns)
data = DataFrame(a = a,a2 = a2,a3=a3,a4=a4,a5=a5, b = b, c = c)

fm = @formula(a ~ b + c)
fm2 = @formula(a2 ~ b + c)
fm3 = @formula(a3 ~ b + c)
fm4 = @formula(a4 ~ b + c)
fm5 = @formula(a5 ~ b + c)
linearRegressor = lm(fm, data)
linearRegressor2 = lm(fm2, data)
linearRegressor3 = lm(fm3, data)
linearRegressor4 = lm(fm4, data)
linearRegressor5 = lm(fm5, data)

readdir("data")

f = open("data/w_optimvar2008.txt", "r")    
line = 0  
 
# read till end of file
while ! eof(f) 
    # read a new / next line for every iteration          
    s = readline(f)         
    line += 1
    println("$line . $s")
end

using DelimitedFiles

w_optimvar = (w_any = readdlm("data/w_optimvar2008.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
w_optimvar_short = (w_any = readdlm("data/w_optimvar2008_short.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)

w_optimcvar95 = (w_any = readdlm("data/w_optimcvar95-2008.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
w_optimcvar95_short = (w_any = readdlm("data/w_optimcvar95-2008_short.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)

w_optimcvar99 = (w_any = readdlm("data/w_optimcvar99-2008.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
w_optimcvar99_short = (w_any = readdlm("data/w_optimcvar99-2008_short.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)

five_γ[1] = (w_any = readdlm("data/w_optimgap"*string(a[1])*"-2008.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
five_γ_short[1] = (w_any = readdlm("data/w_optimgap"*string(a[1])*"-2008_short.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
five_γ[2] = (w_any = readdlm("data/w_optimgap"*string(a[2])*"-2008.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
five_γ_short[2] = (w_any = readdlm("data/w_optimgap"*string(a[2])*"-2008_short.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
five_γ[3] = (w_any = readdlm("data/w_optimgap"*string(a[3])*"-2008.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)
five_γ_short[3] = (w_any = readdlm("data/w_optimgap"*string(a[3])*"-2008_short.txt",',');w_any[1] = parse(Float64,w_any[1][2:end]);w_any[end] = parse(Float64,w_any[end][1:end-1]);w_any = Float64.(vec(w_any));w_any)

using Plots
using Distortions
λs = exp.(range(log(0.001),stop=log(7),length=10))
plot(0.001:0.01:1.0, MAXMINVAR.(0.001:0.01:1.0,λs[1]),label="$(λs[1])")
plot!(0.001:0.01:1.0, MAXMINVAR.(0.001:0.01:1.0,λs[2]),label="$(λs[2])")
plot!(0.001:0.01:1.0, MAXMINVAR.(0.001:0.01:1.0,λs[3]),label="$(λs[3])")
plot!(0.001:0.01:1.0, MAXMINVAR.(0.001:0.01:1.0,λs[4]),label="$(λs[4])")
plot!(0.001:0.01:1.0, MAXMINVAR.(0.001:0.01:1.0,λs[5]),label="$(λs[5])")
plot!(0.001:0.01:1.0, MAXMINVAR.(0.001:0.01:1.0,λs[6]),label="$(λs[6])")





Plots.histogram(Any[indexChangeMPT .- 1,indexChangevar .- 1,indexChangecvar95 .- 1,indexChangecvar99 .- 1] ,fillalpha=0.2)

tableComponents = [indexChangeMPT .- 1,indexChangevar .- 1,indexChangecvar95 .- 1,indexChangecvar99 .- 1, [allRtrns[:,y2k_pos+k:end]'*five_γ[i] for i = 1:6]...]

x = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

kwantielen = [   
    quantile(tableComponents[1],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[2],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[3],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[4],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';    
    quantile(tableComponents[5],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[6],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[7],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[8],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[9],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])';
    quantile(tableComponents[10],[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])']

one = PlotlyJS.scatter(;x=x,    y=kwantielen[1,:], mode="lines",line_shape="hv",name="MPT")
two = PlotlyJS.scatter(;x=x,    y=kwantielen[2,:], mode="lines",line_shape="hv",    name="VaR")
three = PlotlyJS.scatter(;x=x,  y=kwantielen[3,:], mode="lines",line_shape="hv",    name="CVaR 95")
four = PlotlyJS.scatter(;x=x,   y=kwantielen[4,:], mode="lines",line_shape="hv",    name="CVaR 99")
five = PlotlyJS.scatter(;x=x,   y=kwantielen[5,:], mode="lines",line_shape="hv",    name="Conic $(a[1])")
six = PlotlyJS.scatter(;x=x,    y=kwantielen[6,:], mode="lines",line_shape="hv",    name="Conic $(a[2])")
seven = PlotlyJS.scatter(;x=x,  y=kwantielen[7,:], mode="lines",line_shape="hv",    name="Conic $(a[3])")
eight = PlotlyJS.scatter(;x=x,  y=kwantielen[8,:], mode="lines",line_shape="hv",    name="Conic $(a[4])")
nine = PlotlyJS.scatter(;x=x,   y=kwantielen[9,:], mode="lines",line_shape="hv",    name="Conic $(a[5])")
ten = PlotlyJS.scatter(;x=x,    y=kwantielen[10,:], mode="lines",line_shape="hv",   name="Conic $(a[6])")
p = PlotlyJS.plot([one,two,three,four,five,six,seven,eight,nine,ten])

one = PlotlyJS.scatter(;x=sort(tableComponents[1]),y=(1:740)/740, mode="lines",     name="MPT")
two = PlotlyJS.scatter(;x=sort(tableComponents[2]),y=(1:740)/740, mode="lines",     name="VaR")
three = PlotlyJS.scatter(;x=sort(tableComponents[3]),y=(1:740)/740, mode="lines",   name="CVaR 95")
four = PlotlyJS.scatter(;x=sort(tableComponents[4]),y=(1:740)/740, mode="lines",    name="CVaR 99")
five = PlotlyJS.scatter(;x=sort(tableComponents[5]),y=(1:740)/740, mode="lines",    name="Conic $(a[1])")
six = PlotlyJS.scatter(;x=sort(tableComponents[6]),y=(1:740)/740, mode="lines",     name="Conic $(a[2])")
seven = PlotlyJS.scatter(;x=sort(tableComponents[7]),y=(1:740)/740, mode="lines",   name="Conic $(a[3])")
eight = PlotlyJS.scatter(;x=sort(tableComponents[8]),y=(1:740)/740, mode="lines",   name="Conic $(a[4])")
nine = PlotlyJS.scatter(;x=sort(tableComponents[9]),y=(1:740)/740, mode="lines",    name="Conic $(a[5])")
ten = PlotlyJS.scatter(;x=sort(tableComponents[10]),y=(1:740)/740, mode="lines",    name="Conic $(a[6])")
PlotlyJS.plot([one,two,three,four,five,six,seven,eight,nine,ten])

MPT_hist    = kde(tableComponents[1])
var_hist    = kde(tableComponents[2])
cvar95_hist = kde(tableComponents[3])
cvar99_hist = kde(tableComponents[4])
conicone_hist = kde(tableComponents[5])
conictwo_hist = kde(tableComponents[6])
conicthree_hist = kde(tableComponents[7])
conicfour_hist = kde(tableComponents[8])
conicfive_hist = kde(tableComponents[9])
conicsix_hist = kde(tableComponents[10])

onekde =    PlotlyJS.scatter(x=MPT_hist.x,          y=cumsum(MPT_hist.density/sum(MPT_hist.density)),     name="MPT"  , lw=2   )
twokde =    PlotlyJS.scatter(x=var_hist.x,          y=cumsum(var_hist.density/sum(var_hist.density)),         name="VaR"  , lw=2   )
threekde =  PlotlyJS.scatter(x=cvar95_hist.x,       y=cumsum(cvar95_hist.density/sum(cvar95_hist.density)),   name="CVaR 95" , lw=2   )
fourkde =   PlotlyJS.scatter(x=cvar99_hist.x,       y=cumsum(cvar99_hist.density/sum(cvar95_hist.density)),   name="CVaR 99" , lw=2   )
fivekde=    PlotlyJS.scatter(x=conicone_hist.x  ,   y=cumsum(conicone_hist.density/sum(conicone_hist.density)),    name="Conic $(a[1])" , lw=2   )
sixkde=     PlotlyJS.scatter(x=conictwo_hist.x  ,   y=cumsum(conictwo_hist.density/sum(conictwo_hist.density)),    name="Conic $(a[2])" , lw=2   )
sevenkde=   PlotlyJS.scatter(x=conicthree_hist.x,   y=cumsum(conicthree_hist.density/sum(conicthree_hist.density)),  name="Conic $(a[3])" , lw=2   )
eightkde=   PlotlyJS.scatter(x=conicfour_hist.x ,   y=cumsum(conicfour_hist.density)/sum(conicfour_hist.density),   name="Conic $(a[4])" , lw=2   )
ninekde =   PlotlyJS.scatter(x=conicfive_hist.x ,   y=cumsum(conicfive_hist.density)/sum(conicfive_hist.density),   name="Conic $(a[5])" , lw=2   )
tenkde=     PlotlyJS.scatter(x=conicsix_hist.x  ,   y=cumsum(conicsix_hist.density)/sum(conicsix_hist.density),    name="Conic $(a[6])" , lw=2   )
PlotlyJS.plot([onekde,twokde,threekde,fourkde,fivekde,sixkde,sevenkde,eightkde,ninekde,tenkde])