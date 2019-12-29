"""
Code for Dynamic Mode Decomposition (DMD)
and its demonstration by using CFD data
@ author mickn
"""

using Plots, LinearAlgebra, Distributions, Random
Random.seed!(123)

struct DMD
    Δt::Float64     # 時間差分（固定？）CFDもしかしたらIntになってるかも．
    x₀::Vector      # データの初期値
    m::Int          # データ行列の採用時系列数
end

""" 特異値分解を用いたDMDの計算 """
function DMD!(Ψ₀::Matrix, Ψ₁::Matrix)
    F = svd(Ψ₀)
    Fs = adjoint(F.U)*Ψ₁*F.V*inv(F.S)           # サイズの小さい目的関数
    eig = eigen(Fs)
end


""" k番目のデータからの予測 """
predict(Φ::Matrix, Λ::Matrix, dmd::DMD) = Φ*exp(Λ^k) * pinv(Φ) * dmd.x₀

""" 離散系→連続系への変換 """
disc2cont(t::Float64, Φ::Matrix, Λ::Matrix, dmd::DMD) = Φ * exp(t/dmd.Δt .*log(Λ)) * pinv(ϕ)*x₀


function test(N::Int, x::Matrix)
    d=Normal()
    n=rand(d, (N,1))
    return x.+n
end

truemodel(t::Matrix) = 10 .*t

const ts = ones(100,1)

input = test(100,truemodel(ts))

psi1 = input[1:end-1,:]
psi2 = input[2:end,:]

mydmd = DMD(1.,[0.],99)

a = DMD!(psi1, psi2)
