function collect_positions(root::Box{p,T}) where {p,T}
    X = T[]
    for leaf in leaves(root)
        append!(X, position(leaf))
    end
    return reshape(X, ndims(root), length(X)÷ndims(root))
end

function collinear_dims(X, x0)
    issame = X .== x0
    nsame = sum(issame, 1)
    n = length(x0)
    nc = zeros(Int, n)
    for k in Compat.axes(X, 2)
        nsame[1,k] == n-1 || continue
        i = findfirst(x->!x, view(issame, :, k))
        nc[i] += 1
    end
    return nc
end

function collinear_dims(root::Box, box::Box)
    collinear_dims(collect_positions(root), position(box))
end

donothing(args...) = nothing

function generate_randboxes(::Type{B}, n, nboxes, callback=donothing) where B<:Box
    function newx(box, sd)
        bb = boxbounds(box, sd)
        x = bb[1] + (bb[2]-bb[1])*rand()
    end
    nc = CoordinateSplittingPTrees.maxchildren(B)-1
    lower = fill(0.0, n)
    upper = fill(1.0, n)
    pos = fill(1/2, n)
    world = World(lower, upper, pos, rand())
    root = B(world)
    lvs = collect(leaves(root))
    while length(lvs) < nboxes
        box = lvs[rand(1:length(lvs))]
        sd = rand(1:n, degree(B))
        while length(unique(sd)) < degree(B)
            sd = rand(1:n, degree(B))
        end
        xs = ntuple(i->newx(box, sd[i]), degree(B))
        newboxes = Box(box, (sd...,), xs, (rand(nc)...,))
        callback(box, sd, xs, newboxes)
        lvs = collect(leaves(root))
    end
    return root
end

# Store (position(box), boxbounds(box)) tuples for each box
function record_geometry!(data, top, splitdims, xs, newboxes)
    function bitlogical(i)
        l = falses(length(splitdims))
        l.chunks[1] = i
        return l
    end
    b = position(top)
    p = copy(b); p[splitdims] = [xs...]
    bbs = boxbounds(top)
    bbb, bbp = copy(bbs), copy(bbs)
    for sd in splitdims
        mid = (b[sd] + p[sd])/2
        lower, upper = (bbb[sd][1], mid), (mid, bbp[sd][2])
        bbb[sd] = b[sd] < p[sd] ? lower : upper
        bbp[sd] = b[sd] < p[sd] ? upper : lower
    end
    data[getleaf(top)] = (b, bbb)
    for i = 1:length(newboxes)
        xbx = copy(b)
        l = bitlogical(i)
        sds = splitdims[l]
        xbx[sds] = p[sds]
        bbx = copy(bbb)
        bbx[sds] = bbp[sds]
        data[newboxes[i]] = (xbx, bbx)
    end
    return data
end

function dimpair(dimorder::AbstractVector{<:Integer}, p::Integer)
    n = length(dimorder)
    pairings = Union{NTuple{p,Int},NTuple{n%p,Int}}[]
    for i = 0:p:n-1
        push!(pairings, (dimorder[i+1:min(i+p,n)]...,))
    end
    return pairings
end

anydups(t::Tuple{Int}) = false
anydups(t::Tuple{Int,Int}) = t[1] == t[2]
anydups(t::Tuple{Int,Int,Int}) = t[1] == t[2] || t[1] == t[3] || t[2] == t[3]

function testequal_offdiag(Cp, B)
    for I in CartesianRange(size(B))
        if !anydups(I.I)
            @test Cp[I] ≈ B[I]
        end
    end
    return nothing
end

function hasnan_offdiag(Cp)
    for I in CartesianRange(size(Cp))
        !anydups(I.I) && isnan(Cp[I]) && return true
    end
    return false
end

function randpoint_inside(bb::Tuple{T,T}) where T<:Real
    if isfinite(bb[1]) && isfinite(bb[2])
        return bb[1] + (bb[2]-bb[1])*T(0.1+0.8*rand()) # avoid within 10% of edge
    elseif isfinite(bb[1])
        return T(bb[1] + 0.1 + rand())
    elseif isfinite(bb[2])
        return T(bb[2] - 0.1 - rand())
    end
    return T(rand())
end

randpoint_inside(box::Box) = [randpoint_inside(bb) for bb in boxbounds(box)]


function modelvalue(c, g, Q, b, y, prec, x)
    n = length(x)
    val = c + g'*(x - b)
    for j = 1:n, i = j:n
        coef = CoordinateSplittingPTrees.Qcoef_value(i, j, x, b, y, prec)
        if coef != 0  # avoids using NaNs unless we need them
            val += Q[i,j]*coef*(1 + (i!=j))  # off-diagonals are done once
        end
    end
    return val
end

function modelvalue_chi(c, g, Q, b, y, prec, x)
    n = length(x)
    val = c + g'*(x - b)
    for j = 1:n, i = 1:n
        coef = (x[i] - b[i]) * (x[j] - chi(j, i, b, y, prec))/2
        if coef != 0  # avoids using NaNs unless we need them
            val += Q[i,j]*coef
        end
    end
    return val
end

function checkfit(c, g, Q, b, y, prec, box::Box{2})
    n = ndims(box)
    box, success = CoordinateSplittingPTrees.chaintop(box)
    success || error("no chain found at $box")
    for i = 1:ceil(Int, n/2)
        split = box.split
        mv = modelvalue(c, g, Q, b, y, prec, position(split.self))
        isnan(mv) || @assert(value(split.self) ≈ mv)
        for ic = 1:3
            bx = split.others.children[ic]
            mv = modelvalue(c, g, Q, b, y, prec, position(bx))
            isnan(mv) || @assert(value(bx) ≈ mv)
        end
        box = split.others.children[end]
    end
    return nothing
end
