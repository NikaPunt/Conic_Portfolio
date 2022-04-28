# Returns for the ordered vector R the distorted expectation defined by distortion Ψ
# result == ψ(m/M) - Ψ((m-1)/M)    where m = 1,...,M and M is length(R)
function bid(R::Vector{Float64},ΨminusΨ::Vector{Float64})
    # ΨminusΨ = [Ψs[m]-Ψs[m-1] for m=1:M]
    # println("M = ", M, ",\nR = ",size(R),",\nΨminusΨ = ",size(ΨminusΨ))
    return ΨminusΨ' * R
end

# Reward μ_p = -∑ᵢ(aᵢ⋅bid(Rᵢ))
function REWARD(w::Vector{Float64},Rs::Matrix{Float64},ΨminusΨ::Vector{Float64})
    μₚ = -w' * [bid(sort(Rs[1,:]),ΨminusΨ) for i = 1:length(w)]
    return μₚ
end

# Reward μ_p = -∑ᵢ(aᵢ⋅bid(Rᵢ))
function REWARDSORTED(w::Vector{Float64},Rs::Matrix{Float64},ΨminusΨ::Vector{Float64})
    μₚ = -w' * [    bid(Rs[i,:],ΨminusΨ) for i = 1:length(w)]
    return μₚ
end

# Risk ̃c(a) = -bid(Rₚ)

#             w = n×1 vector     Rs = n×m matrix
function RISK(w::Vector{Float64},Rs::Matrix{Float64},ΨminusΨ::Vector{Float64})
    Rₚ = Rs' * w
    c = -bid(sort(Rₚ),ΨminusΨ)
    return c
end 

# using ProfileView: @profview #loading this makes everything so slow
# vektor = rand(5);vektor=vektor/sum(vektor);matriks=rand(5,1000);
# @profview (for i = 1:10000; RISK(vektor,matriks);end)

function GAP(w::Vector{Float64},Rs::Matrix{Float64},ΨminusΨ::Vector{Float64})
    c = RISK(w,Rs,ΨminusΨ)
    μₚ = REWARD(w,Rs,ΨminusΨ)
    return μₚ - c
end

function GAPSORTED(w::Vector{Float64},Rs::Matrix{Float64},RsSorted::Matrix{Float64},ΨminusΨ::Vector{Float64})
    c = RISK(w,Rs,ΨminusΨ)
    μₚ = REWARD(w,RsSorted,ΨminusΨ)
    return μₚ - c
end