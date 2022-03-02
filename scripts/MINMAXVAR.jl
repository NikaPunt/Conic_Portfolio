function fastlog2(x::Float32)::Float32
    y = Float32(reinterpret(Int32, x))
    y *= 1.1920928955078125f-7
    y - 126.94269504f0
end
function fastlog2(x::Float64)::Float32
   fastlog2(Float32(x))
end

# https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastexp.h
function fastpow2(x::Float32)::Float32
    clipp = x < -126.0f0 ? -126.0f0 : x
    clipp = min(126f0, max(-126f0, x))
    reinterpret(Float32, UInt32((1 << 23) * (clipp + 126.94269504f0)))
end
function fastpow2(x::Float64)::Float32
   fastpow2(Float32(x))
end

# https://github.com/etheory/fastapprox/blob/master/fastapprox/src/fastpow.h
function fastpow(x::Real, y::Real)::Real
    fastpow2(y * fastlog2(x))
end

# n = 100000000
# as = rand(n)
# bs = rand(n)
# c1 = zeros(n)
# c2 = zeros(n)
# c3 = zeros(n)
# @time (for i = 1:n; c1[i]=as[i]^bs[i];end)
# @time (for i = 1:n; c2[i]=exp(bs[i]*log(as[i]));end)
# @time (for i = 1:n; c3[i]=fastpow(as[i],bs[i]);end)
# [c1, c2, c3]
# norm(x::Vector{Float64}) = sqrt(sum(x.^2))
# println("Relative error of a^b = exp(log(a)*b) is ", norm(c1-c2)/norm(c2))
# println("Relative error of a^b = fastpow2(fastlog2(a)*b) is ", norm(c1-c3)/norm(c3))


function MINMAXVAR(u,λ) # tiny bit faster than minmaxvar2 and the precision error is almost zero.
    minmaxvar = 1-exp(log((1-exp(log(u)*(1/(λ+1)))))*(λ+1))
    return minmaxvar
end

function MINMAXVAR2(u::Float64,λ::Float64)::Float64
    minmaxvar = 1-(1-u^(1/(λ+1)))^(λ+1)
    return minmaxvar
end

function MINMAXVAR3(u,λ) #fastest but least precise
    minmaxvar = 1-fastpow(1-fastpow(u,1/(λ+1)),λ+1)
    return minmaxvar
end

# @time (for i = 1:1; c1[i]=MINMAXVAR(as[i],bs[i]);end)
# @time (for i = 1:1; c2[i]=MINMAXVAR2(as[i],bs[i]);end)
# @time (for i = 1:1; c3[i]=MINMAXVAR3(as[i],bs[i]);end)

# @time (for i = 1:n; c1[i]=MINMAXVAR(as[i],bs[i]);end)
# @time (for i = 1:n; c2[i]=MINMAXVAR2(as[i],bs[i]);end)
# @time (for i = 1:n; c3[i]=MINMAXVAR3(as[i],bs[i]);end)

# norm(c1-c2)/norm(c2)
# norm(c2-c3)/norm(c2)

# c3