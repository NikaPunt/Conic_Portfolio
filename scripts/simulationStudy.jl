include("HeaderFile.jl")
println("Importing datasets")
using Dates
begin
    filenames = readdir("data/StudyStocks")
    N = length(filenames)-1
    uniqueNames = Vector{String}(undef,N)
    for i = 1:N
        name = filenames[i]
        coin = name[1:end-4]
        uniqueNames[i] = coin
    end



    df_list = Vector{DataFrame}(undef,N)
    for i = 1:N
        df_list[i] = DataFrame(CSV.File("data/StudyStocks/"*filenames[i],delim=","))
    end

    #Let's get all the dates for BTC (longest)
    dates = df_list[2]."Date"
    for i = 1:N
        dates = intersect(dates,df_list[i]."Date")
    end

    for i = 1:N # make sure we have all dates
        df_list[i] = filter(row -> row.:Date in dates, df_list[i])
    end
end

