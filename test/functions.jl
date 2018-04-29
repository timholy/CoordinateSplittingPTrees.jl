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
    splits = [(1/2,3/4) for i = 1:n]
    world = World(lower, upper, splits, rand())
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
