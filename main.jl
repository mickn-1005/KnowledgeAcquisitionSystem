using Plots, DataFrames, CSV, Random, StatsBase
using TSne
# using ScikitLearn
Random.seed!(123)

const N_SAMPLE = 500
const CATEGORY = [1,2,4,5,7,8,9]

const filepath = "data/noname/191119_HK_real_svtlm.csv"

tlm = CSV.read(filepath)
tlm = dropmissing(tlm)      # 欠損地を含む行をデータから削除
tlmnm = names(tlm)

"""前処理として，Stringで読み込まれたIDを全て番号に変換"""
const MODE_CATEGORY = unique(tlm[!, tlmnm[4]])
function enum_modeid(tlm, tlmnm)
    for nm in tlmnm
        if eltype(tlm[!, nm])==String
            catlist = unique(tlm[!, nm])
            tlm[!, nm] = [findfirst(==(x), catlist) for x in tlm[!, nm]]    # ラベルを数値に変換．
        end
    end
    return tlm
end
tlm = enum_modeid(tlm, tlmnm)
const ENUM_MC = unique(tlm[!, tlmnm[4]])

"""モードを限定したDataFrameの作成"""
function delete_category(tlm)
    for id in CATEGORY
        tlm = tlm[findall(!=(ENUM_MC[id]), tlm[4]),:]
    end
    return tlm
end
tlm = delete_category(tlm)

"""DataFrameを列に関して正規化"""
function df_scale(df)
    d = describe(df, :mean, :std)
    for i in length(size(df)[2])
        df[i] = (df[i] .- d[i,3]) / d[i,2]
    end
    return df
end
tlm = df_scale(tlm)

"""ランダムにDataFrameからデータ抽出"""
minibatch(tlm, n=N_SAMPLE) = tlm[sample(1:size(tlm)[1], n, replace=false), :]
testset = minibatch(tlm)

# 説明変数の列（説明変数のPruningを乱択するため）
nmarr = sample(tlmnm, length(tlmnm), replace=false)

# ind = 1:size(testset)[2]
# ind = 7:size(testset)[2]
ind = 55:size(testset)[2]
df_tsne(testset, indices) = tsne(Matrix(testset[indices]))
pred = DataFrame(df_tsne(testset, ind))

"""結果の評価として，各クラスタの重心（2d前提）を計算"""
cog(pred, tlm, mcind) = describe(pred[findall(==(ENUM_MC[mcind]), tlm[!, 4]),:], :mean)

"""重心に対してまとまりの良さを平均２乗誤差で判定"""
euc_msqerr_(pred, tlm, mdid, cog, xy) = (pred[findall(==(mdid), tlm[!, 4]), xy] .- cog[xy,2]).^2
euc_msqerr(pred, tlm, mdid, cog) = sum(euc_msqerr_(pred, tlm, mdid, cog, 1)) + sum(euc_msqerr_(pred, tlm, mdid, cog, 2))
[euc_msqerr(pred, testset, mdid, cog(pred, testset, mdid)) for mdid in unique(testset[4])]

function tsne_scat2(pred, tlm, indices)
    scatter(pred[findall(==(indices[1]), tlm[!, 4]), 1], 
                pred[findall(==(indices[1]), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                # label=mode_category[1]
                )
    [scatter!(pred[findall(==(id), tlm[!, 4]), 1],
                pred[findall(==(id), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                # label=mode_category[id]
                ) for id in indices[2:end]][end]        # 配列の最後をReturnすることによって，全てが描写されたPlotを返せる．
end
function tsne_scat3()
    scatter(pred[findall(==(ENUM_MC[1]), tlm[!, 4]), 1],
                pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]),2],
                pred[findall(==(unique(numerized_mc)[1]), tlm[!, 4]),3],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                )
    [scatter!(pred[findall(==(ENUM_MC[id]), tlm[!, 4]), 1],
                pred[findall(==(ENUM_MC[id]), tlm[!, 4]),2],
                pred[findall(==(ENUM_MC[id]), tlm[!, 4]),3],
                markeralpha = 0.5,
                markerstrokealpha = 0.,) for id in 2:9][end]
end
tsne_scat2(pred, testset, unique(testset[4]))
# tsne_scat3()

"""main"""
# CSVの読み込みとデータの整形