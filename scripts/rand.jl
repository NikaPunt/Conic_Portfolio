# using Pkg
# Pkg.add("GLM")
using GLM
a = vec( five_γ_short[1]'*Rtrns)
a2 = vec(five_γ_short[2]'*Rtrns)
a3 = vec(five_γ_short[3]'*Rtrns)
a4 = vec(five_γ_short[4]'*Rtrns)
a5 = vec(five_γ_short[5]'*Rtrns)
b = vec(w_optimMPT_short'*Rtrns)
c = vec(w_optimcvar_short'*Rtrns)
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
