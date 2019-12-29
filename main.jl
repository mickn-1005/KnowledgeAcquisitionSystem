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

function enum_modeid(tlm, tlmnm)
    for nm in tlmnm
        if eltype(tlm[!, nm])==String
            catlist = unique(tlm[!, nm])
            tlm[!, nm] = [findfirst(==(x), catlist) - 1 for x in tlm[!, nm]]    # ラベルを数値に変換．
        end
    end
    return tlm
end
tlm = enum_modeid(tlm, tlmnm)
numerized_mc = tlm[!, tlmnm[4]]

function df_scale(df)
    d = describe(df, :mean, :std)
    for i in length(size(df)[2])
        df[i] = (df[i] .- d[i,3]) / d[i,2]
    end
    return df
end

tlm = df_scale(tlm)

"""
説明変数のみを格納したDataFrameの作成
"""
express = Matrix(tlm[7:end])
pred = tsne(express)
# pred = tsne(express, 3)

function tsne_scat2()
    scatter(pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]), 1], 
                pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                )
    [scatter!(pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]), 1],
                pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,) for id in 2:9][end]
end
function tsne_scat3()
    scatter(pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]), 1], 
                pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]),2],
                pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]),3],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                )
    [scatter!(pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]), 1],
                pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]),2],
                pred[findall(==(unique(numerized_mc)[id]), tlm[!, 4]),3],
                markeralpha = 0.5,
                markerstrokealpha = 0.,) for id in 2:9][end]
end
tsne_scat2()
# tsne_scat3()

