include("HeaderFile.jl")

# Extract coin names from file names
crypto_filenames = readdir("datasets/crypto10")
N = length(crypto_filenames)
crypto_names = Vector{String}(undef,N)
for i = 1:N
    name = crypto_filenames[i]
    coin = name[9:end-10]
    crypto_names[i] = coin
end

dflist = Vector{DataFrame}(undef,N)
for i = 1:N
    dflist[i] = DataFrame(CSV.File("datasets/crypto10/"*crypto_filenames[i],delim=",",header=2))
end

#Let's get all the dates for BTC (longest)
dates = dflist[2]."date"
for i = 1:N
    dates = intersect(dates,dflist[i]."date")
end

for i = 1:N
    dflist[i] = filter(row -> row.:date in dates, dflist[i])
end
