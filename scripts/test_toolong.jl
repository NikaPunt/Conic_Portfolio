function sleep5(a)
    sleep(5.0)
    return "Done"
end

time_from_now(seconds) = round(Int, 10^9 * seconds + time_ns())
function runtime_limiter(f::Function, args...; timeout=60)
    t = @async f(args)
    end_time = time_from_now(timeout)
    result = missing
    while time_ns() <= end_time
        sleep(0.1)
        if istaskdone(t)
            result = fetch(t)
            break
        end
    end
    return result
end

# Missing...
runtime_limiter(sleep5,3;timeout=4)

# Done...
runtime_limiter(sleep5,3;timeout=5)
