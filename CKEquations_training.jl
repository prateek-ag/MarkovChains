# Load X and y variable
using JLD

# Load initial probabilities and transition probabilities of Markov chain
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])

# Set 'k' as number of states
k = length(p1)

# Confirm that initial probabilities sum up to 1
@assert sum(p1) == 1.0

# Confirm that transition probabilities sum up to 1, starting from each state
sums = sum(pt,dims=2)
for j in 1:length(sums)
    @assert isapprox(sums[j], 1.0)
end

include("misc.jl")

# Define sampling function for Markov chain
function sampleAncestral(d, p, T)
    k = length(p)
    @assert size(T)[1] == k
    @assert size(T)[2] == k
    x = zeros(Int64, d)
    for j in 1:d
        if j == 1
            # initial case. Do integer conversion to 
            # make futher indexing simple
            x[j] = convert(Int64, sampleDiscrete(p))
        else
            # sample transitions from state at j-1
            x[j] = convert(Int64, sampleDiscrete(T[:,x[j-1]]))
        end
    end
    return x
end

using Printf

# # Do Monte-Carlo on the loaded data
# d = 50
# iters = 10000
# counts = zeros(k) # number of times we saw state k at end
# for i in 1:iters
#     sequence = sampleAncestral(d, p1, pt')
#     counts[sequence[d]] = counts[sequence[d]] + 1
# end
# marginalsMC = zeros(k)
# for i in 1:k
#     marginalsMC[i] = counts[i]/iters
# end
# @show marginalsMC
# @show sum(marginalsMC)

# CK Equations
function marginalCK(d, p, T)
    # Given Markov assumption, Tj
    # is the same for all j, so just T
    k = length(p)
    π = zeros(d,k)
    π[1,:] = p
    for j in 2:d
        π[j,:] = T*π[j-1,:]
    end
    return π
end

# marginalsCK = marginalCK(50, p1, pt')
# @show marginalsCK[50,:]
# @show sum(marginalsCK[50,:])
# @show marginalsCK
# @show mapslices(argmax, marginalsCK, dims=2)

function viterbiDecode(d, p, T)
    k = length(p)
    M = zeros(d,k) # result table
    B = zeros(Int64,d,k) # backtrack table
    # 1. Set initial optimal sub-structure
    M[1,:] = p
    # 2. Compute optimal results up to d
    for j in 2:d
        for s in 1:k
            p = [T[s,x_s]*M[j-1,x_s] for x_s in 1:k]
            opt = argmax(p)
            M[j,s] = p[opt]
            B[j,s] = opt
        end
    end
    # 3. Backtrack
    x = zeros(Int64, d)
    x[d] = argmax(M[d,:])
    for j in d-1:-1:1
        x[j] = B[j+1,x[j+1]]
    end
    return x
end

function joint(x, p, T)
    P = p[x[1]]
    for j in 2:length(x)
        P *= T[x[j],x[j-1]]
    end
    return P
end

function bfDecode(d, p, T)
    k = length(p)
    pBest = 0.0
    xBest = zeros(d)
    x = ones(Int64, d)
    count = 0
    while x[1] <= k
        # 1. check
        px = joint(x,p,T)
        if px >= pBest
            xBest = copy(x)
            pBest = px
        end
        # 2. increment
        x[d] += 1
        for j in d:-1:2
            if x[j] > k
                x[j] = 1
                x[j-1] += 1
            end
        end
        count += 1
    end
    @assert count == k^d
    return xBest
end

pDoom = [0.4 0.35 0.25 0.0 0.0]
TDoom = [
    0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0;
    1.0 0.0 0.0 0.0 0.0;
    0.0 1.0 1.0 0.0 0.0;
]

# test brute force solution
@assert bfDecode(2, pDoom, TDoom) == [1,4]

# test with some small values against BF
@assert viterbiDecode(2, pDoom, TDoom) == bfDecode(2, pDoom, TDoom)
@assert viterbiDecode(4, p1, pt') == bfDecode(4, p1, pt')

# show results
@show viterbiDecode(50, p1, pt')
@show viterbiDecode(100, p1, pt')

# assume start in grad school.
pStart = zeros(length(p1))
pStart[3] = 1.0
@show marginalCK(50, pStart, pt')[50,:]


# Rejection sampling
samples = 10000
accepted = 0
rejected = 0
counts = zeros(k) # number of times we saw state k at end
for i in 1:samples
    sequence = sampleAncestral(10, p1, pt')
    if sequence[10] == 6
        global accepted += 1
        counts[sequence[5]] += 1
    else
        global rejected += 1
    end
end
marginalsMC = zeros(k)
for i in 1:k
    marginalsMC[i] = counts[i]/accepted
end
@show accepted
@show rejected
@show marginalsMC
@show sum(marginalsMC)

# Forward-backward algorithm for Markov chain
# solving exact p(x_j = c | x_10 = 6)
T = pt'         # transition matrix
k = length(p1)  # number of states
J = 10          # known time conditioned on
vJ = 6          # known state conditioned on
d = J           # we need up to time J at least
M = zeros(d,k)  # forward messages
V = zeros(d,k)  # backward messages
# Run algorithm
# 1. Initialize messages with the initial probabilities
M[1,:] = p1
# 2. Forward
for j in 2:d
    if j == J
        # set message with known conditioned
        # all other values are 0
        M[j,vJ] = 1
    else
        for xj in 1:k
            M[j,xj] = sum([T[xj,x_s]*M[j-1,x_s] for x_s in 1:k])
        end
    end
end
# 3. Backward
for j in d:-1:1
    if j == J
        V[j,vJ] = 1
    else
        for xj in 1:k
            V[j,xj] = sum([T[x_s,xj]*V[j+1,x_s] for x_s in 1:k])
        end
    end
end
# 4. Compute probability. Can be used for any
# j,c but we want j=5
j = 5
P = zeros(k)
for c in 1:k
    P[c] = M[j,c]*V[j,c]
end
s = (1/sum(P)) # scaling factor
P = s*P        # final result
@show P