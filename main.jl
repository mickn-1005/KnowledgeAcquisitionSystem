using Plots, DataFrames, CSV, Random, StatsBase#, ProgressBar
using TSne
# using ScikitLearn
Random.seed!(123)

const N_SAMPLE = 500
const CATEGORY = [1,2,7,8,9]

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
            tlm[!, nm] = [Float64(findfirst(==(x), catlist)) for x in tlm[!, nm]]    # ラベルを数値に変換．
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

minibatch(tlm, n=N_SAMPLE) = tlm[sample(1:size(tlm)[1], n, replace=false), :]

# const EXPL = 1:size(testset)[2]
# const EXPL = 7:size(testset)[2]
# const EXPL = 55:size(testset)[2]
const EXPL = [i for i in 55:size(tlm)[2]]

testset = minibatch(tlm)

df_tsne(testset) = DataFrame(tsne(Matrix(testset),2,0,10000,45., progress=false))

pred = df_tsne(testset[EXPL])

"""結果の評価として，各クラスタの重心（2d前提）を計算"""
cog(pred, tlm, mcind) = describe(pred[findall(==(mcind), tlm[!, 4]),:], :mean)

"""重心に対してまとまりの良さを平均２乗誤差で判定"""
euc_msqerr_(pred, tlm, mdid, cog, xy) = (pred[findall(==(mdid), tlm[!, 4]), xy] .- cog[xy,2]).^2
euc_msqerr(pred, tlm, mdid, cog) = sum(euc_msqerr_(pred, tlm, mdid, cog, 1)) + sum(euc_msqerr_(pred, tlm, mdid, cog, 2))
evalf(pred, tlm) = sum([euc_msqerr(pred, tlm, mdid, cog(pred, tlm, mdid)) for mdid in unique(tlm[4])])
# evalfmean(pred,tlm) = sum([euc_msqerr(pred, tlm, mdid, cog(pred, tlm, mdid)) for mdid in unique(tlm[4])])

evalf(pred, testset)

function pruning_expl(tlm, ind)
    # 説明変数の列（説明変数のPruningを乱択するため）
    nmarr = sample(EXPL, length(EXPL), replace=false)
    for nmid in nmarr
        """ランダムにDataFrameからデータ抽出"""
        testset = minibatch(tlm)
        # println(nmid)
        ind = sort(union(ind, [nmid]))

        n_itr = 3
        try
            w = mean([evalf(df_tsne(testset[ind]), testset) for i in 1:n_itr])
            wo =mean([evalf(df_tsne(testset[symdiff(ind, nmid)]), testset) for i in 1:n_itr])
            if w > wo
                println("drop id:  ", tlmnm[nmid])
                ind = symdiff(ind, nmid)
            end
        catch
            println("cannot convert t-sne...")
            continue
        end
    end
    # println("validate...")
    return df_tsne(tlm), ind
end

function randompruning()
    epoch = 4
    result, ind = pruning_expl(tlm, EXPL)
    for t in 1:epoch
        println(t, " th epoch. number of explanation parameter = ", length(ind))
        result, ind = pruning_expl(tlm, ind)
    end
    return result, ind
end

function tsne_scat2(pred, tlm, indices)
    scatter(pred[findall(==(indices[1]), tlm[!, 4]), 1],
                pred[findall(==(indices[1]), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                # label=mode_category[1]
                label=MODE_CATEGORY[indices[1]],
                legend=:outertopright)
    [scatter!(pred[findall(==(id), tlm[!, 4]), 1],
                pred[findall(==(id), tlm[!, 4]),2],
                markeralpha = 0.5,
                markerstrokealpha = 0.,
                label=MODE_CATEGORY[id]
                ) for id in indices[2:end]][end]        # 配列の最後をReturnすることによって，全てが描写されたPlotを返せる．
end

 r,i = randompruning()
# r1,i1 = randompruning()
# r2,i2 = randompruning()
tsne_scat2(r, tlm, unique(Int.(tlm[4])))
title!("sum of squared error, perplex = 45.")
tsne_scat2(pred, testset, unique(testset[4]))

tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,10000,30.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 30.,  iteration= 10000")

tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,10000,5.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 5.,  iteration= 10000")
tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,30000,50.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 50.,  iteration= 30000")
tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,30000,30.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 30.,  iteration= 30000")
tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,30000,5.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 5.,  iteration= 30000")

tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,10000,40.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 40.,  iteration= 10000")
tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,30000,40.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 40.,  iteration= 30000")

tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,10000,35.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 35.,  iteration= 10000")
tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,10000,45.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 45.,  iteration= 10000")

tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,30000,35.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 35.,  iteration= 30000")
tsne_scat2(DataFrame(tsne(Matrix(tlm[EXPL]), 2,0,30000,45.)), tlm, unique(Int.(tlm[4])))
title!("perplex = 45.,  iteration= 30000")

const zeuspath = "data/noname/191122_ZEUS_real_svtlm.csv"
zeus = CSV.read(zeuspath)
zeus = dropmissing(zeus)      # 欠損地を含む行をデータから削除
zeusnm = names(zeus)
zeus = enum_modeid(zeus, zeusnm)
zeus = df_scale(zeus)
zeustsne = df_tsne(zeus[1408:end,34:end])