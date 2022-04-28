include("HeaderFile.jl")

struct Asset
    name::String    
    returns::Vector{Float64}
    mean::Float64
    vol::Float64
end

println("Importing datasets")
begin
    uniqueNames = ["BRK-B","NKE","FB","V","GOOGL"];
    csv_brk = CSV.File("datasets/stocks/BRK-B.csv");
    csv_nke = CSV.File("datasets/stocks/NKE.csv");
    csv_fb = CSV.File("datasets/stocks/FB.csv");
    csv_v = CSV.File("datasets/stocks/V.csv");
    csv_googl = CSV.File("datasets/stocks/GOOGL.csv");

    df_brk = DataFrame(csv_brk) # NIKE(NKE) GOOGLE(GOOGL) VISA(V) META(FB) BERKSHIRE(BRK-A)
    df_nke = DataFrame(csv_nke)
    df_fb = DataFrame(csv_fb)
    df_v = DataFrame(csv_v)
    df_googl = DataFrame(csv_googl)

    # list of all stock dataframes.
    df_list = [df_brk, df_nke, df_fb, df_v, df_googl];

    # list of all stock dividend yields.
    yields = [0, 0.0078, 0, 0.0069, 0];
end

println("Calculating daily returns")
begin
    nrAssets = length(df_list); #number of assets
    sampleReturns = zeros(nrAssets,length(df_brk."Adj Close")-1) #matrix containing the daily returns of each asset in each row.
    AssetArray = Array{Asset}(undef,nrAssets);
    for i = 1:nrAssets
        df = df_list[i]
        name = uniqueNames[i]
        closes = df."Adj Close"
        returns = log.(closes[2:end]./closes[1:end-1])
        gemiddelde = 20*mean(returns)
        volatiliteit = sqrt(20)*std(returns)
        AssetArray[i] = Asset(name,returns,gemiddelde,volatiliteit)
    end
end

means(a::Vector{Asset}) = Vector{Float64}([ass.mean for ass in a])
vols(a::Vector{Asset}) = Vector{Float64}([ass.vol for ass in a])
returns(a::Vector{Asset}) = Vector{Vector{Float64}}([ass.returns for ass in a])

#Make sure to run this at least once
println("Calculating Optimal Mean-Variance Portfolio")
begin
    μ = means(AssetArray);
    Σ = 20*cov(hcat(returns(AssetArray)...));
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