include("HeaderFile.jl")

println("Importing datasets")
begin
    names = ["BRK-B","NKE","FB","V","GOOGL"];
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
    assetShiftedReturns = Array{Float64,2}(undef,nrAssets,length(df_brk."Adj Close")-1); #same as up here but then making mean = 0

    for i = 1:nrAssets
        df = df_list[i]
        closes = df."Adj Close"
        returns = log.(closes[2:end]./closes[1:end-1])
        sampleReturns[i,:] = returns
        gemiddelde = mean(returns)
        shiftedReturns = returns-repeat([gemiddelde],length(returns))
        assetShiftedReturns[i,:] = shiftedReturns;
    end
end

assetMeans = mapslices(mean,sampleReturns,dims=2)
A = [assetMeans'; ones(nrAssets)']
lengte = 100
# b = [range(minimum(assetMeans),stop=maximum(assetMeans),length=lengte)';ones(lengte)']
b = [range(minimum(assetMeans),stop=maximum(assetMeans),length=lengte)';ones(lengte)']
ws = zeros(5,lengte)
τs = zeros(lengte)

println("Starting optimization")
β = 0.95
q = size(sampleReturns,2)
mincvar_model = Model();
set_optimizer(mincvar_model, Ipopt.Optimizer)
set_optimizer_attribute(mincvar_model, "constr_viol_tol", 1e-15)
set_optimizer_attribute(mincvar_model, "acceptable_tol", 1e-15)
@variable(mincvar_model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
@variable(mincvar_model, τ)
@variable(mincvar_model, u[1:q] >= 0)
@objective(mincvar_model, Min, τ+sum(u)/(1-β));
@constraint(mincvar_model, con2, u' .>= -(w'*sampleReturns .+ τ)/q)
@constraint(mincvar_model, con, A[2,:]'*w == 1);
optimize!(mincvar_model);
mincvar_weights = value.(w)
mincvar_var = value.(τ)
mincvar = objective_value(mincvar_model)
delete(mincvar_model,con);
delete.(mincvar_model,con2);
unregister(mincvar_model,:con);
unregister(mincvar_model,:con2);


cvar_efficient = zeros(lengte)
β = 0.95
q = size(sampleReturns,2)
model = Model();
set_optimizer(model, Ipopt.Optimizer)
set_optimizer_attribute(model, "constr_viol_tol", 1e-15)
set_optimizer_attribute(model, "acceptable_tol", 1e-15)
set_optimizer_attribute(model, "print_level", 0)
@variable(model, w[1:nrAssets] >= 0) # you can unregister w through unregister(model, w)
@variable(model, τ)
@variable(model, u[1:q] >= 0)
@objective(model, Min, τ+sum(u)/(1-β));
@constraint(model, con2, u' .>= -(w'*sampleReturns .+ τ)/q)
for i = 1:lengte
    # println("Thread number ",Threads.threadid()," working on iteration ",i,"/",lengte)
    println(i);
    @constraint(model, con, A*w .== b[:,i]);
    optimize!(model);
    ws[:,i] = value.(w)
    τs[i] = value.(τ)
    cvar_efficient[i] = objective_value(model);
    delete(model,con);
    unregister(model,:con);
end

plot(cvar_efficient, b[1,:])