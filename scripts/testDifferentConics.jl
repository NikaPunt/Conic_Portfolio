dataList[345]

y2k_pos = dataList[345]["y2k_pos"]
assetIndices = dataList[345]["assetIndices"]

assetdfs = df_list[assetIndices]

assetcloses = [assetdfs[i]."Close"[y2k_pos-1000:y2k_pos+63] for i = 1:9]
assetreturns = [(assetcloses[i][2:end] .- assetcloses[i][1:end-1]) ./ assetcloses[i][1:end-1] for i = 1:9]
XX = vcat([assetreturns[i][1:1000]' for i = 1:9]...)

assetShiftedRtrns = zeros(9,length(assetreturns[1])-63)
for i = 1:9
    assetShiftedRtrns[i,:] = assetreturns[i][1:1000] .- mean(assetreturns[i][1:1000])
end

Rtrns = simulateJointReturns(assetShiftedRtrns, 10000)

a_double = vec([0.14 0.4 1.0 2.6 5.0 0.14 0.4 1.0 2.6 5.0])
all_γ3 = Vector{Vector{Float64}}(undef,10)
@time Threads.@threads for i = 1:10
    id = Threads.threadid()
    println("Conic $(a_double[i]) on thread $id on iteration $i")
    nummer = a_double[i] 
    if i < 6
        all_γ3[i] = getMinConicWeights(XX,Rtrns,a_double[i],false)
    else
        all_γ3[i] = getMinConicWeights(XX,Rtrns,a_double[i],true)
    end
end

Rtrns2 = simulateJointReturns(assetShiftedRtrns, 100000)

a_double = vec([0.14 0.4 1.0 2.6 5.0 0.14 0.4 1.0 2.6 5.0])
all_γ2 = Vector{Vector{Float64}}(undef,10)
Threads.@threads for i = 1:10
    id = Threads.threadid()
    println("Conic $(a_double[i]) on thread $id on iteration $i")
    nummer = a_double[i] 
    if i < 6
        all_γ2[i] = getMinConicWeights(Rtrns2,a_double[i],false)
    else
        all_γ2[i] = getMinConicWeights(Rtrns2,a_double[i],true)
    end
end

println("our new all gamma")
println(all_γ)
println("Other all gamma")
println(all_γ2)
println("Last all gamma")
println(all_γ3)

all_γ[10]
all_γ2[10]
all_γ3[10]
