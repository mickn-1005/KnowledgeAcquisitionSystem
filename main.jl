using Plots, DataFrames, CSV#, ScikitLearn

filepath = "data/dat41/191121_HK_real_svtlm.csv"

tlm = CSV.read(filepath)
tlmnm = names(tlm)

k = unique(eltype.(eachcol(tlm)))

tlm[!,tlmnm[1]]

mode_category = tlm[!, tlmnm[5]]

for nm in tlmnm
    if eltype(tlm[!, nm])==String
        catlist = unique(tlm[!, nm])
        tlm[!, nm] = [findfirst(==(x), catlist) - 1 for x in tlm[!, nm]]    # ラベルを数値に変換．
    end
end

a = [1,2,3]

k = tlm[!, a]
