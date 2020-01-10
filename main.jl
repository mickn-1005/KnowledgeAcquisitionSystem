using Plots, DataFrames, CSV, Random, StatsBase#, ProgressBar
using LinearAlgebra, Statistics, Distances, ProgressMeter
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

function Hbeta!(P::AbstractVector, D::AbstractVector, beta::Number)
    @inbounds P .= exp.(-beta .* D)
    sumP = sum(P)
    H = log(sumP) + beta * dot(D, P) / sumP
    @inbounds P .*= 1/sumP
    return H
end

function perplexities(D::AbstractMatrix{T}, tol::Number = 1e-5, perplexity::Number = 30.0;
                      max_iter::Integer = 50,
                      progress::Bool=true) where T<:Number
    (issymmetric(D) && all(x -> x >= 0, D)) ||
        throw(ArgumentError("Distance matrix D must be symmetric and positive"))
    # 初期化
    n = size(D, 1)
    P = fill(zero(T), n, n)
    beta = fill(one(T), n) 
    Htarget = log(perplexity) 
    Di = fill(zero(T), n)
    Pcol = fill(zero(T), n)

    progress && (pb = Progress(n, "Computing point perplexities"))
    for i in 1:n
        progress && update!(pb, i)

        betai = 1.0
        betamin = 0.0
        betamax = Inf

        copyto!(Di, view(D, :, i))
        Di[i] = prevfloat(Inf)
        minD = minimum(Di) 
        @inbounds Di .-= minD

        H = Hbeta!(Pcol, Di, betai)
        Hdiff = H - Htarget

        tries = 0
        while abs(Hdiff) > tol && tries < max_iter
            if Hdiff > 0.0
                betamin = betai
                betai = isfinite(betamax) ? (betai + betamax)/2 : betai*2
            else
                betamax = betai
                betai = (betai + betamin)/2
            end

            H = Hbeta!(Pcol, Di, betai)
            Hdiff = H - Htarget
            tries += 1
        end

        @inbounds P[:, i] .= Pcol
        beta[i] = betai
    end
    progress && finish!(pb)
    # Return final P-matrix
    return P, beta
end

function pca(X::AbstractMatrix, ndims::Integer = 50)
    (n, d) = size(X)
    (d <= ndims) && return X
    Y = X .- mean(X, dims=1)
    C = Symmetric((Y' * Y) ./ (n-1))
    Ceig = eigen(C, (d-ndims+1):d)
    return Y * reverse(Ceig.vectors, dims=2)
end

kldivel(p, q) = ifelse(p > zero(p) && q > zero(q), p*log(p/q), zero(p))

pairwisesqdist(X::AbstractMatrix, dist::Bool) =
    dist ? X.^2 : pairwise(SqEuclidean(), X')

pairwisesqdist(X::AbstractVector, dist::Function) =
    [dist(x, y)^2 for x in X, y in X]

pairwisesqdist(X::AbstractMatrix, dist::Function) =
    [dist(view(X, i, :), view(X, j, :))^2 for i in 1:size(X, 1), j in 1:size(X, 1)]

pairwisesqdist(X::AbstractMatrix, dist::SemiMetric) =
    pairwise(dist, X').^2

function tsne(X::Union{AbstractMatrix, AbstractVector}, ndims::Integer = 2, reduce_dims::Integer = 0,
              max_iter::Integer = 1000, perplexity::Number = 30.0;
              distance::Union{Bool, Function, SemiMetric} = false,
              min_gain::Number = 0.01, eta::Number = 200.0, pca_init::Bool = false,
              initial_momentum::Number = 0.5, final_momentum::Number = 0.8, momentum_switch_iter::Integer = 250,
              stop_cheat_iter::Integer = 250, cheat_scale::Number = 12.0,
              progress::Bool=true,
              )
    # preprocess X
    ini_Y_with_X = false
    if isa(X, AbstractMatrix) && (distance !== true)
        ndims < size(X, 2) || throw(DimensionMismatch("X has fewer dimensions ($(size(X,2))) than ndims=$ndims"))

        ini_Y_with_X = true
        X = X * (1.0/std(X)::eltype(X)) # note that X is copied
        if 0<reduce_dims<size(X, 2)
            reduce_dims = max(reduce_dims, ndims)
            X = pca(X, reduce_dims)
        end
    end
    n = size(X, 1)
    # Initialize embedding
    if pca_init && ini_Y_with_X
        if reduce_dims >= ndims
            Y = X[:, 1:ndims] # reuse X PCA
        else
            @assert reduce_dims <= 0 # no X PCA
            Y = pca(X, ndims)
        end
    else
        Y = randn(n, ndims)
    end

    dY = fill!(similar(Y), 0)     # gradient vector
    iY = fill!(similar(Y), 0)     # momentum vector
    gains = fill!(similar(Y), 1)  # how much momentum is affected by gradient

    # Compute P-values
    D = pairwisesqdist(X, distance)
    P, beta = perplexities(D, 1e-5, perplexity, progress=progress)
    P .+= P' # make P symmetric
    P .*= cheat_scale/sum(P) # normalize + early exaggeration
    sum_P = cheat_scale

    # Run iterations
    progress && (pb = Progress(max_iter, "Computing t-SNE"))
    Q = fill!(similar(P), 0)     # temp upper-tri matrix with 1/(1 + (Y[i]-Y[j])²)
    Ymean = similar(Y, 1, ndims) # average for each embedded dimension
    sum_YY = similar(Y, n, 1)    # square norms of embedded points
    L = fill!(similar(P), 0)     # temp upper-tri matrix for KLdiv gradient calculation
    Lcolsums = similar(L, n, 1)  # sum(Symmetric(L), 2)
    last_kldiv = NaN
    for iter in 1:max_iter
        # Compute pairwise affinities
        BLAS.syrk!('U', 'N', 1.0, Y, 0.0, Q) # Q=YY', updates only the upper tri of Q
        @inbounds for i in 1:size(Q, 2)
            sum_YY[i] = Q[i, i]
        end
        sum_Q = 0.0
        @inbounds for j in 1:size(Q, 2)
            sum_YYj_p1 = 1.0 + sum_YY[j]
            Qj = view(Q, :, j)
            Qj[j] = 0.0
            for i in 1:(j-1)
                sqdist_p1 = sum_YYj_p1 - 2.0 * Qj[i] + sum_YY[i]
                @fastmath Qj[i] = ifelse(sqdist_p1 > 1.0, 1.0 / sqdist_p1, 1.0)
                sum_Q += Qj[i]
            end
        end
        sum_Q *= 2 # the diagonal and lower-tri part of Q is zero

        # Compute the gradient
        inv_sum_Q = 1.0 / sum_Q
        fill!(Lcolsums, 0.0) # column sums
        # fill the upper triangle of L (gradient)
        @inbounds for j in 1:size(L, 2)
            Lj = view(L, :, j)
            Pj = view(P, :, j)
            Qj = view(Q, :, j)
            Lsumj = 0.0
            for i in 1:(j-1)
                @fastmath Lj[i] = l = (Pj[i] - Qj[i]*inv_sum_Q) * Qj[i]
                Lcolsums[i] += l
                Lsumj += l
            end
            Lcolsums[j] += Lsumj
        end
        @inbounds for (i, ldiag) in enumerate(Lcolsums)
            L[i, i] = -ldiag
        end
        # dY = -4LY
        BLAS.symm!('L', 'U', -4.0, L, Y, 0.0, dY)

        # Perform the update
        momentum = iter <= momentum_switch_iter ? initial_momentum : final_momentum
        @inbounds for i in eachindex(gains)
            gains[i] = max(ifelse((dY[i] > 0) == (iY[i] > 0),
                                  gains[i] * 0.8,
                                  gains[i] + 0.2),
                           min_gain)
            iY[i] = momentum * iY[i] - eta * (gains[i] * dY[i])
            Y[i] += iY[i]
        end
        @inbounds Y .-= mean!(Ymean, Y)

        # stop cheating with P-values
        if sum_P != 1.0 && iter >= min(max_iter, stop_cheat_iter)
            P .*= 1/sum_P
            sum_P = 1.0
        end
        # Compute the current value of cost function
        if !isfinite(last_kldiv) || iter == max_iter ||
            (progress && mod(iter, max(max_iter÷20, 10)) == 0)
            local kldiv = 0.0
            @inbounds for j in 1:size(P, 2)
                Pj = view(P, :, j)
                Qj = view(Q, :, j)
                kldiv_j = 0.0
                for i in 1:(j-1)
                    # P and Q are symmetric (only the upper triangle used)
                    @fastmath kldiv_j += kldivel(Pj[i], Qj[i])
                end
                kldiv += 2*kldiv_j + kldivel(Pj[j], Q[j])
            end
            last_kldiv = kldiv/sum_P + log(sum_Q/sum_P) # adjust wrt P and Q scales
        end
        progress && update!(pb, iter,
                            showvalues = Dict(:KL_divergence => @sprintf("%.4f%s", last_kldiv,
                                                                         iter <= stop_cheat_iter ? " (warmup)" : "")))
    end
    progress && (finish!(pb))

    return Y
end

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