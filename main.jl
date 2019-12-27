using Plots, DataFrames, CSV, ScikitLearn

filepath = "data/dat41/191121_HK_real_svtlm.csv"

# CSV.validate(filepath)

tlm = CSV.read(filepath)
tlmnm = names(tlm)

# k = eltypes(tlm)
# for types in k
#     if Missing == types     # 欠損値が含まれないか確認
#         println("hoge")
#     end
# end
# unique(k)

for i in tlmnm
    println(i)
end

tlm[!,tlmnm[1]]

mode_category = unique(tlm[!, tlmnm[4]])

for nm in tlmnm
    if eltype(tlm[nm])==String
        catlist = unique(tlm[!, nm])
        tlm[!, nm] = [findfirst(==(x), catlist) - 1 for x in tlm[!, nm]]    # ラベルを数値に変換．
    end
end

