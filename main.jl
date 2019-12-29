using Plots, DataFrames, CSV
using TSne
# using ScikitLearn

filepath = "data/noname/191119_HK_real_svtlm.csv"

tlm = CSV.read(filepath)
tlm = dropmissing(tlm)      # 欠損地を含む行をデータから削除
tlmnm = names(tlm)


k = unique(eltype.(eachcol(tlm)))

tlm[!,tlmnm[1]]

mode_category = tlm[!, tlmnm[4]]

for nm in tlmnm
    if eltype(tlm[!, nm])==String
        catlist = unique(tlm[!, nm])
        tlm[!, nm] = [findfirst(==(x), catlist) - 1 for x in tlm[!, nm]]    # ラベルを数値に変換．
    end
end

numerized_mc = tlm[!, tlmnm[4]]

"""
説明変数のみを格納したDataFrameの作成
"""
express = Matrix(tlm[7:end])
pred = tsne(express)

# scatter(pred[:,1], pred[:,2])

function t_sne_scat()
    scatter(pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]), 1], pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                )
    [scatter!(pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]), 1],
                pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,) for id in 2:9][end]
end
t_sne_scat()

