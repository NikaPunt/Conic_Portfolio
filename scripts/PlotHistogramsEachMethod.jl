using KernelDensity

MPT_long_hist   = kde(vec(w_optimMPT'*Rtrns))
var_long_hist   = kde(vec(w_optimvar'*Rtrns))
cvar_long_hist  = kde(vec(w_optimcvar'*Rtrns))
gap_long_hist   = kde(vec(w_optimgap'*Rtrns))
MPT_short_hist  = kde(vec(w_optimMPT_short'*Rtrns))
var_short_hist  = kde(vec(w_optimvar_short'*Rtrns))
cvar_short_hist = kde(vec(w_optimcvar_short'*Rtrns))
gap_short_hist  = kde(vec(w_optimgap_short'*Rtrns))

halfnhalf = kde(vec((w_optimcvar_short+w_optimMPT_short)'*Rtrns)/2)

plot(MPT_short_hist.x,MPT_short_hist.density,   label="Long-Short MPT"  , lw=2   )
plot!(var_short_hist.x,var_short_hist.density,  label="Long-Short VaR"  , lw=2   )
plot!(cvar_short_hist.x,cvar_short_hist.density,label="Long-Short CVaR" , lw=2   )
plot!(gap_short_hist.x,gap_short_hist.density,  label="Long-Short Conic", lw=2   )
plot!(halfnhalf.x,halfnhalf.density,            label="Long-Short Half-MPT Half-CVaR", lw=2   )


conicweights = vcat([twentyiterations[i][1]' for i = 1:5]...)
γs = [0.1,3.725,6.45,9.625,12.8]
open("data/weights.txt", "w") do file
    write(file, "Weights for conic max gap portfolios:\n$conicweights\nwith γ values:\n$γs\n\n\nMPT weights (long, long-short)\n$w_optimMPT\n$w_optimMPT_short\n\nVaR weights (long, long-short)\n$w_optimvar\n$w_optimvar_short\n\nCVaR weights (long, long-short)\n$w_optimcvar\n$w_optimcvar_short")
end

# With Conic iterations
kdes = [kde_lscv(vec(twentyiterations[i][1]'*Rtrns)) for i = 1:5]
plot(kdes[1].x,kdes[1].density, label="Conic λ=0.1", lw=1,          ls=:auto)
plot!(kdes[2].x,kdes[2].density, label="Conic λ=3.275", lw=1.25,    ls=:auto)
plot!(kdes[3].x,kdes[3].density, label="Conic λ=6.45", lw=1.5,      ls=:auto)
plot!(kdes[4].x,kdes[4].density, label="Conic λ=9.625", lw=1.75,    ls=:auto)
plot!(kdes[5].x,kdes[5].density, label="Conic λ=12.8", lw=2,        ls=:auto)

[sum(-quantile(vec(twentyiterations[i][1]'*Rtrns),range(0,stop=0.05,length=1000)))/1000 for i = 1:5]

histogram(vec(twentyiterations[1][1]'*Rtrns),density=true)
histogram(Any[vec(twentyiterations[5][1]'*Rtrns), vec(twentyiterations[1][1]'*Rtrns)], line=(1,0.2,:green), fillcolor=[:red :black], fillalpha=0.1)
