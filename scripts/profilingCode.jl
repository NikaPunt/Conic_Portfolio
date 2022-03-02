using Profile

function myfunc()
    i = 1
    println("Thread number ",Threads.threadid()," working on iteration ",i,"/",length(I))
    μₚ = I[i]
    function con_c!(c, x)
        c[1] = sum(x)-1
        c[2] = REWARDSORTED(x,ReturnsSorted)-μₚ
        c
    end
    lc = [0,0]; uc = [0,0];
    x0 = rand(N)
    x0 = x0/sum(x0)
    df = TwiceDifferentiable(funRISK, x0)
    dfc = TwiceDifferentiableConstraints(con_c!, lx, ux, lc, uc)

    res = optimize(df, dfc, x0, IPNewton())
    w = Optim.minimizer(res)
    ws[i,:] = w
end

myfunc()

@profile myfunc()

Profile.print()

x = 1
const y = x
x+=1
println(y)