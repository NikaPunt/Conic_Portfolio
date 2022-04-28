include("HeaderFile.jl")

struct Asset
    name::String    
    returns::Vector{Float64}
    mean::Float64
    vol::Float64
end

println("Importing datasets")
begin
    crypto_filenames = readdir("datasets/crypto10")
    N = length(crypto_filenames)
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
    sampleRtrns = zeros(nrAssets,length(df_list[1]."close")-1) #matrix containing the daily rtrns of each asset in each row.
    AssetArray = Array{Asset}(undef,nrAssets);
    for i = 1:nrAssets
        df = df_list[i]
        name = uniqueNames[i]
        closes = df."close"
        rtrns = log.(closes[2:end]./closes[1:end-1])
        gemiddelde = 30.437*mean(rtrns)
        volatiliteit = sqrt(30.437)*std(rtrns)
        AssetArray[i] = Asset(name,rtrns,gemiddelde,volatiliteit)
    end
end

means(a::Vector{Asset}) = Vector{Float64}([ass.mean for ass in a])
vols(a::Vector{Asset}) = Vector{Float64}([ass.vol for ass in a])
returns(a::Vector{Asset}) = Vector{Vector{Float64}}([ass.returns for ass in a])

#Make sure to run this at least once
println("Calculating Optimal Mean-Variance Portfolio")
begin
    μ = means(AssetArray);
    Σ = 30.437*cov(hcat(returns(AssetArray)...));
    model = Model();
    set_optimizer(model, Ipopt.Optimizer)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(model, "acceptable_tol", 1e-15)
    @variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
    @constraint(model, lt1, sum(w) == 1);
    @objective(model, Min, w'*Σ*w);
    optimize!(model);
    w_optimMPT = value.(w)
    σ_minimum = objective_value(model)
    μ_minimum = w_optimMPT'*μ
    println("Best Mean-Variance Weights: $w_optimMPT \nWith Variance: $σ_minimum")
end


println("Calculating Efficient MPT Frontier")
begin
    # let us discretize a bunch of values for μ_required
    μ_required = range(minimum(μ),stop=maximum(μ),length=100);
    σ_efficient = zeros(100);

    model_frontier = Model();
    set_optimizer(model_frontier, Ipopt.Optimizer)
    set_optimizer_attribute(model_frontier, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(model_frontier, "acceptable_tol", 1e-15)
    set_optimizer_attribute(model_frontier, "print_level", 0)
    @variable(model_frontier, w[1:nrAssets] >= 0) # you can unregister w through unregister(model_frontier, w)
    @constraint(model_frontier, lt1, sum(w) == 1);
    @objective(model_frontier, Min, w'*Σ*w);

    for i = 1:100
        println(i);
        μ_i = μ_required[i];
        @constraint(model_frontier, con, transpose(w)*μ == μ_i);
        optimize!(model_frontier);
        σ_efficient[i] = objective_value(model_frontier);
        delete(model_frontier,con);
        unregister(model_frontier,:con);
    end
end

println("Calculating Optimal Mean-Variance Long-Short Portfolio")
begin
    μ = means(AssetArray);
    Σ = 30.437*cov(hcat(returns(AssetArray)...));
    model = Model();
    set_optimizer(model, Ipopt.Optimizer)
    set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
    set_optimizer_attribute(model, "acceptable_tol", 1e-15)
    @variable(model, w[1:nrAssets]) # you can unregister w through unregister(model, w)
    @constraint(model, lt1, sum(w) == 1);
    @objective(model, Min, w'*Σ*w);
    optimize!(model);
    w_optimMPT_short = value.(w)
    σ_minimum_short = objective_value(model)
    μ_minimum_short = w_optimMPT'*μ
    println("Best Mean-Variance Weights: $w_optimMPT_short \nWith Variance: $σ_minimum_short")
end
