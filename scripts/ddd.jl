function RISK(w::Vector{Float64},Rs::Matrix{Float64})
    Rₚ = Rs' * w
    c = -bid(sort(Rₚ))
    return c
end 

function RISK2(w::Vector{Float64},Rs::Matrix{Float64})
    Rₚ = vec(w' * Rs)
    c = -bid(sort(Rₚ))
    return c
end 

@time RISK(vektor, matriks)
@time RISK2(vektor, matriks)