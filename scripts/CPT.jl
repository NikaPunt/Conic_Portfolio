include("HeaderFile.jl")

println("Importing datasets")
begin
    crypto_filenames = readdir("datasets/crypto10")
    const N = length(crypto_filenames)
    uniqueNames = Vector{String}(undef,N)
    for i = 1:N
        name = crypto_filenames[i]
        coin = name[9:end-10]
        uniqueNames[i] = coin
    end

    df_list = Vector{DataFrame}(undef,N)
    for i = 1:N
        df_list[i] = DataFrame(CSV.File("datasets/crypto10/"*crypto_filenames[i],delim=",",header=2))
    end

    #Let's get all the dates for BTC (longest)
    dates = df_list[2]."date"
    for i = 1:N
        dates = intersect(dates,df_list[i]."date")
    end

    for i = 1:N
        df_list[i] = filter(row -> row.:date in dates, df_list[i])
    end
end

println("Calculating daily returns")
begin
    nrAssets = length(df_list); #number of assets
    sampleRtrns = zeros(nrAssets,length(df_list[1]."close")-1) #matrix containing the daily returns of each asset in each row.
    assetShiftedRtrns = Array{Float64,2}(undef,nrAssets,length(df_list[1]."close")-1); #same as up here but then making mean = 0

    for i = 1:nrAssets
        df = df_list[i]
        closes = df."close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        sampleRtrns[i,:] = rtrns
        gemiddelde = mean(rtrns)
        shiftedRtrns = rtrns-repeat([gemiddelde],length(rtrns))
        assetShiftedRtrns[i,:] = shiftedRtrns;
    end
end

println("Starting Independent Component Analysis")
begin
    ica_comps = ICA_assets(assetShiftedRtrns);
    # X = A*S      where X are the returns, A is mixing matrix and S is independent comps
    # S = W'*X
    W = ica_comps.W;
    A = ica_comps.mixing;
    S = ica_comps.indcomps; #these are the independent components. For each row, we need σ, ν, θ. 
end

println("Extracting implied VG parameters on independent components")
pars = get_params_timeseries_returns(S);

const M = 10000
println("Simulating ",M," monthly returns of independent components")
begin
    V = VG_Params2MRet([getVGParams(pars[i,:]...,1,20) for i = 1:nrAssets], M)
end

println("Transforming independent returns to joint returns")
begin
    #returns Y = A*V where A is the mixing matrix.
    Y = A*V;
    const N = size(Y,1);
    Rtrns′ = zeros(N,M);
    # Later on we also need an already sorted Rtrns matrix (sorted rows)
    RtrnsSorted′ = zeros(N,M)
    # Let us center the returns in Y 
    for i = 1:N
        avgSumExp = 1/M*sum(exp.(Y[i,:]))
        # println(avgSumExp)
        for m = 1:M
            Rtrns′[i,m] = exp(Y[i,m])-avgSumExp
        end
        RtrnsSorted′[i,:] = sort(Rtrns′[i,:])
    end

    Rtrns = Rtrns′
    RtrnsSorted = RtrnsSorted′
end

include("CPT_Optimization.jl")
twentyiterations = Vector{Tuple{Vector{Float64},Float64}}(undef,5)
a = range(0.1,stop=12.8,length=5)
short_weights = MPT_Optimization(Rtrns)[1]
for i = 1:5
    twentyiterations[i] = CPT_Optimization(Rtrns,a[i],(true,short_weights))
end

GAP(twentyiterations[5][1],Rtrns,(Ψs = [MAXMINVAR(m/M,12.8) for m=0:M];
ΨminusΨ = [Ψs[m]-Ψs[m-1] for m=2:(M+1)];ΨminusΨ))



##### Optional: calculate efficient frontiers #####
# begin ## Long only optimization preamble    
#     W_id = Matrix{Float64}(LinearAlgebra.I,N,N); # Identity matrix
#     rewardStocks = [REWARD(W_id[i,:],Rtrns) for i = 1:N];
#     I = range(minimum(rewardStocks);stop=maximum(rewardStocks),length=20);
#     ws = zeros(length(I),N);
# end

# begin ## Long only efficient frontier calculation
#     funRISK(x) = RISK(x,Rtrns)

#     # @threads for i = 1:length(I)
#     @threads for i = 1:20
#         println("Thread number ",Threads.threadid()," working on iteration ",i,"/",length(I))
#         μₚ = I[i]
#         # x = [1/2, 1/2]
#         # c[1] = sum(x)-1 == 1 - 1 == 0 
#         # c[2] = REWARDSORTED(x,RtrnsSorted) - μₚ
#         function con_c!(c, x) 
#             c[1] = sum(x)-1
#             c[2] = REWARDSORTED(x,RtrnsSorted)-μₚ
#             c
#         end
#         lc = [0,0]; uc = [0,0];
#         x0 = rand(N)
#         x0 = x0/sum(x0)
#         df = TwiceDifferentiable(funRISK, x0)
#         dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

#         res = optimize(df, dfc, x0, IPNewton())
#         w = Optim.minimizer(res)
#         ws[i,:] = w
#     end
# end

# begin ## Plot the frontier
#     plot(   [RISK(ws[i,:],Rtrns) for i = 1:length(I)], 
#             [REWARD(ws[i,:],Rtrns) for i = 1:length(I)],
#             linecolor=:blue,
#             label="Conic Efficient Frontier",
#             title="Efficient Frontier",
#             lw=3    )
#     plot!(  [0,1],
#             [gap_optim, gap_optim+1],
#             label="Conic max diversification line",
#             lw=3    )
#     plot!(  [RISK(w_optimgap,Rtrns)],
#             [REWARD(w_optimgap,Rtrns)],
#             seriestype=:scatter,
#             label="Max diversified portfolio",
#             markershape=:diamond,
#             markersize=6    )
#     # plot!(  [RISK(w_optimMPT,Rtrns)],
#             # [REWARD(w_optimMPT,Rtrns)],
#             # seriestype=:scatter,
#             # label="Minimum variance portfolio",
#             # markershape=:star4,
#             # markersize=6    )
#     # plot!(  [RISK(w_optimSharpe,Rtrns)],
#     #         [REWARD(w_optimSharpe,Rtrns)],
#     #         seriestype=:scatter,
#     #         label="Maximum Sharpe Portfolio",
#     #         markershape=:star5,
#     #         markersize=6    )
#     # ↑↑ comment this out if you haven't calculated the weights through MPT optimization
#     W_id = Matrix{Float64}(LinearAlgebra.I,N,N)
#     plot!(  [RISK(W_id[i,:],Rtrns) for i = [1:3;5:N]], 
#             [REWARD(W_id[i,:],Rtrns) for i = [1:3;5:N]],
#             seriestype = :scatter,
#             label="Stocks",
#             series_annotations=text.(names[[1:3;5:N]], :top,:left;rotation=-20.2,pointsize=8),
#             markercolor=:purple )
#     plot!(  [RISK(W_id[4,:],Rtrns)], 
#             [REWARD(W_id[4,:],Rtrns)],
#             seriestype = :scatter,
#             label=false,
#             series_annotations=text.(names[4], :bottom,:right;rotation=-20.2,pointsize=8),
#             markercolor=:purple )
#     sample_ports = [(a=rand(N);a=a/sum(a);a) for i = 1:5]
#     plot!(  [RISK(sample_ports[i],Rtrns) for i = 1:5],
#             [REWARD(sample_ports[i],Rtrns) for i = 1:5],
#             seriestype=:scatter,
#             label="Randomly assembled portfolios",
#             markeralpha=0.5,
#             markerstrokealpha=0.5   )
#     plot!(title="Conic Efficient Frontier",xlabel="Risk c̃(a)",ylabel="Reward μₚ")
#     plot!(legend=:bottomright)
#     #γ=0.1
#     plot!(xlims=(0,0.0167),ylims=(0,0.02))
#     #γ=0.8
#     # plot!(xlims=(0,0.03),ylims=(0,0.035))
# end


# begin ## Long-Short optimization preamble
#     # The following two lines are already imported in the long only preamble
#     # W_id = Matrix{Float64}(LinearAlgebra.I,N,N) # Identity matrix
#     # rewardStocks = [REWARD(W_id[i,:],Rtrns) for i = 1:N] # Rewards for holding onto one stock

#     # As our returns are no longer limited by the minimum and maximum of rewardStocks
#     # we are going to make our range a little wider than what we had for the variable I
#     δ_reward = minimum(rewardStocks)/1.5 
#     I2 = range(δ_reward; stop=δ_reward+maximum(rewardStocks),length=20)
#     ws2 = zeros(length(I2),N)
# end

# begin ## Long-short efficient frontier calculation
#     funRISK(x) = RISK(x,Rtrns)

#     @threads for i = 1:length(I2)
#         println("Thread number ",Threads.threadid()," working on iteration ",i,"/",length(I2))
#         μₚ = I2[i]
#         function con_c!(c, x)
#             c[1] = sum(x)-1
#             c[2] = REWARD(x,Rtrns)-μₚ
#             c[3] = sqrt(x'*Σ_indcomp_1m*x)
#             c
#         end
#         lc = [0,0,0]; uc = [0,0,Q*1.5/√(12)];

#         x_0 = ws[i,:]

#         df = TwiceDifferentiable(funRISK, x_0)
#         dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

#         res = optimize(df, dfc, x_0, IPNewton())
#         w = Optim.minimizer(res)
#         ws2[i,:] = w
#     end
# end

# begin ## Plot the frontier
#     bitvek = (([sum(ws2[i,:]) for i = 1:20].>0.9) .* ([sum(ws2[i,:]) for i = 1:20].<1.1))
#     ws2 = ws2[bitvek,:]
#     plot([RISK(ws2[i,:],Rtrns) for i = 1:size(ws2,1)], [REWARD(ws2[i,:],Rtrns) for i = 1:size(ws2,1)],linecolor=:blue,label="Conic Efficient Frontier",title="Efficient Frontier",lw=3)
#     plot!([0,1],[gap_optim2, gap_optim2+1],label="Conic max diversification line",lw=3)
#     plot!([RISK(w_optimgap2,Rtrns)],[REWARD(w_optimgap2,Rtrns)],seriestype=:scatter,label="Max diversified portfolio",markershape=:diamond,markersize=6)
#     # plot!([RISK(w_optimMPT,Rtrns)],[REWARD(w_optimMPT,Rtrns)],seriestype=:scatter,label="Minimum variance portfolio",markershape=:star4,markersize=6)
#     # ↑↑ comment this out if you haven't calculated the weights through MPT optimization
#     W_id = Matrix{Float64}(LinearAlgebra.I,N,N)
#     plot!(
#         [RISK(W_id[i,:],Rtrns) for i = [1:3;5:N]], 
#         [REWARD(W_id[i,:],Rtrns) for i = [1:3;5:N]],
#         seriestype = :scatter,label="Stocks",
#         series_annotations=text.(names[[1:3;5:N]], :top,:left;rotation=-20.2,pointsize=8),
#         markercolor=:purple)
#     plot!(
#         [RISK(W_id[4,:],Rtrns)], 
#         [REWARD(W_id[4,:],Rtrns)],
#         seriestype = :scatter,
#         label=false,
#         series_annotations=text.(names[4], :bottom,:right;rotation=-20.2,pointsize=8),
#         markercolor=:purple)
#     sample_ports = [(a=rand(N);a=a/sum(a);a) for i = 1:5]
#     plot!([RISK(sample_ports[i],Rtrns) for i = 1:5],[REWARD(sample_ports[i],Rtrns) for i = 1:5],seriestype=:scatter,label="Randomly assembled portfolios",markeralpha=0.5,markerstrokealpha=0.5)
#     plot!(title="Conic Efficient Frontier",xlabel="Risk c̃(a)",ylabel="Reward μₚ")
#     plot!(legend=:bottomright)
#     #γ=0.1
#     plot!(xlims=(0,0.02),ylims=(0,0.02))
#     #γ=0.8
#     # plot!(xlims=(0,0.03),ylims=(0,0.035))
# end
