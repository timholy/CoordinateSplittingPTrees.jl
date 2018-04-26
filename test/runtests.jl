using CoordinateSplittingPTrees
using CoordinateSplittingPTrees: addpoint!
using Base.Test

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

@testset "Geometry and iteration, CS1" begin
    # For comparing boxbounds
    myapproxeq(t1::Tuple{Real,Real}, t2::Tuple{Real,Real}) = t1[1] ≈ t2[1] && t1[2] ≈ t2[2]
    myapproxeq(v1::Vector, v2::Vector) = all(myapproxeq.(v1, v2))

    # 1-d
    world = World([0], [1], [(0,1)], 10)
    @test eltype(world.lower) == Float64
    root = @inferred(Box{1}(world))
    @test_throws ErrorException Box(root, 1, -0.5, 20)  # out-of-bounds
    @test_throws ErrorException Box(root, 1,  1.5, 20)  # out-of-bounds
    @test_throws ErrorException Box(root, 1,    0, 20)  # coincides with parent
    box = @inferred(Box(root, 1, 1, 20))[1]
    @test @inferred(meta(box)) == 20
    @test @inferred(position(box)) == [1]
    @test @inferred(position(box, 1)) == 1
    @test_throws BoundsError position(box, 2)
    @test @inferred(boxbounds(box)) == [(0.5,1)]
    @test @inferred(boxbounds(box, 1)) == (0.5,1)
    @test_throws BoundsError boxbounds(box, 2)
    @test CoordinateSplittingPTrees.boxscale(box) == [1]
    @test CoordinateSplittingPTrees.epswidth(boxbounds(box, 1)) == eps()
    @test CoordinateSplittingPTrees.epswidth(boxbounds(getleaf(root), 1)) == eps(0.5)
    # evaluation point is at the boundary of parents
    @test_throws AssertionError Box(root, 1, 0.5, 30)  # can only split leaves
    @test_throws ErrorException Box(getleaf(root), 1, 0.5, 30)  # can't eval at upper edge
    box1 = Box(box, 1, 0.5, 30)[1]
    @test boxbounds(box1, 1) == (0.5, 0.75)
    @test boxbounds(getleaf(box), 1) == (0.75, 1.0)
    @test boxbounds(getleaf(root), 1) == (0.0, 0.5)
    @test boxbounds(box1) == [(0.5, 0.75)]
    @test boxbounds(getleaf(box)) == [(0.75, 1.0)]
    @test boxbounds(getleaf(root)) == [(0.0, 0.5)]
    leaf = find_leaf_at(root, [0.3])
    @test isleaf(leaf)
    @test leaf == getleaf(root)
    leaf = find_leaf_at(root, [0.5])
    @test isleaf(leaf)
    @test leaf === box1
    leaf = find_leaf_at(root, [0.8])
    @test isleaf(leaf)
    @test leaf === getleaf(box)
    @test_throws ErrorException find_leaf_at(root, [1.2])
    @test length(root) == 5
    @test length(leaves(root)) == 3

    world = World([0], [Inf], [(0,1)], 10) # with infinite size
    root = Box{1}(world)
    box = Box(root, 1, 1, 20)[1]
    @test boxbounds(box) == [(0.5,Inf)]
    @test boxbounds(box, 1) == (0.5,Inf)
    @test CoordinateSplittingPTrees.boxscale(box) == [1]
    @test CoordinateSplittingPTrees.epswidth(boxbounds(box, 1)) == eps(0.5)
    @test CoordinateSplittingPTrees.epswidth(boxbounds(getleaf(root), 1)) == eps(0.5)

    geom = Dict()
    root = generate_randboxes(Box{1}, 1, 10, (args...)->record_geometry!(geom, args...))
    for leaf in leaves(root)
        pos, bbs = geom[leaf]
        @test position(leaf) == pos
        @test boxbounds(leaf) == bbs
    end

    # 2-d
    world = World([0,-Inf], [Inf,Inf], [(1,2), (1,2)], nothing)
    root = @inferred(Box{1}(world))
    box1 = @inferred(Box(root, 1, 2, nothing))[1]
    box2 = Box(getleaf(root), 2, 2, nothing)[1]

    b = getleaf(root)
    @test @inferred(position(b, 1)) == position(b, 2) == 1
    @test @inferred(position(b)) == [1,1]
    @test @inferred(boxbounds(b, 1)) == (0,1.5)
    @test boxbounds(b, 2) == (-Inf,1.5)
    @test @inferred(boxbounds(b)) == [(0,1.5), (-Inf,1.5)]

    b = box2
    @test position(b, 1) == 1
    @test position(b, 2) == 2
    @test position(b) == [1,2]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf)]

    b = box1
    @test position(b, 1) == 2
    @test position(b, 2) == 1
    @test position(b) == [2,1]
    @test boxbounds(b, 1) == (1.5,Inf)
    @test boxbounds(b, 2) == (-Inf,Inf)  # never split along 2
    @test boxbounds(b) == [(1.5,Inf), (-Inf,Inf)]

    b = getleaf(root)
    @test b.parent.split.dims == (2,)
    b2 = Box(b, 1, 1/3, nothing)[1]
    @test position(b2, 1) ≈ 1/3
    @test position(b2, 2) == 1
    @test position(b2) ≈ [1/3, 1]
    @test myapproxeq(boxbounds(b2, 1), (0, 2/3))
    @test boxbounds(b2, 2) == (-Inf,1.5)
    @test myapproxeq(boxbounds(b2), [(0, 2/3), (-Inf,1.5)])
    b1 = b.split.self
    @test position(b1, 1) == 1
    @test position(b1, 2) == 1
    @test position(b1) == [1, 1]
    @test myapproxeq(boxbounds(b1, 1), (2/3, 1.5))
    @test boxbounds(b1, 2) == (-Inf,1.5)
    @test myapproxeq(boxbounds(b1), [(2/3, 1.5), (-Inf,1.5)])
    b2_2 = Box(b2, 2, -1, nothing)[1]
    @test position(b2_2, 1) ≈ 1/3
    @test position(b2_2, 2) ≈ -1
    @test position(b2_2) ≈ [1/3, -1]
    @test myapproxeq(boxbounds(b2_2, 1), (0, 2/3))
    @test boxbounds(b2_2, 2) == (-Inf,0)
    @test myapproxeq(boxbounds(b2_2), [(0, 2/3), (-Inf,0)])
    b2_1 = b2.split.self
    @test position(b2_1, 1) == position(b2, 1)
    @test position(b2_1, 2) == position(b2, 2)
    @test position(b2_1) == position(b2)
    @test myapproxeq(boxbounds(b2_1, 1), (0, 2/3))
    @test boxbounds(b2_1, 2) == (0,1.5)
    @test myapproxeq(boxbounds(b2_1), [(0, 2/3), (0,1.5)])
    b1_2 = Box(b1, 2, -1, nothing)[1]
    @test position(b1_2, 1) == 1
    @test position(b1_2, 2) == -1
    @test position(b1_2) == [1, -1]
    @test myapproxeq(boxbounds(b1_2, 1), (2/3, 1.5))
    @test boxbounds(b1_2, 2) == (-Inf, 0)
    @test myapproxeq(boxbounds(b1_2), [(2/3, 1.5), (-Inf,0)])
    b1_1 = b1.split.self
    @test position(b1_1, 1) == position(b1, 1)
    @test position(b1_1, 2) == position(b1, 2)
    @test position(b1_1) == position(b1)
    @test myapproxeq(boxbounds(b1_1, 1), (2/3, 1.5))
    @test boxbounds(b1_1, 2) == (0, 1.5)
    @test myapproxeq(boxbounds(b1_1), [(2/3, 1.5), (0, 1.5)])

    geom = Dict()
    root = generate_randboxes(Box{1}, 2, 10, (args...)->record_geometry!(geom, args...))
    for leaf in leaves(root)
        pos, bbs = geom[leaf]
        @test position(leaf) == pos
        @test boxbounds(leaf) == bbs
    end

    # 5-d
    geom = Dict()
    root = generate_randboxes(Box{1}, 5, 20, (args...)->record_geometry!(geom, args...))
    for leaf in leaves(root)
        pos, bbs = geom[leaf]
        @test position(leaf) == pos
        @test boxbounds(leaf) == bbs
    end
end

@testset "Geometry and iteration, CS2" begin
    myapproxeq(t1::Tuple{Real,Real}, t2::Tuple{Real,Real}) = t1[1] ≈ t2[1] && t1[2] ≈ t2[2]
    myapproxeq(v1::Vector, v2::Vector) = all(myapproxeq.(v1, v2))

    # 2-d
    world = World([0,-Inf], [Inf,Inf], [(1,2), (1,2)], nothing)
    root = @inferred(Box{2}(world))
    boxes1 = @inferred(Box(root, (1,2), (2,2), (nothing, nothing, nothing)))

    b = getleaf(root)
    @test @inferred(position(b, 1)) == position(b, 2) == 1
    @test @inferred(position(b)) == [1,1]
    @test @inferred(boxbounds(b, 1)) == (0,1.5)
    @test boxbounds(b, 2) == (-Inf,1.5)
    @test @inferred(boxbounds(b)) == [(0,1.5), (-Inf,1.5)]

    b = boxes1[2]
    @test position(b, 1) == 1
    @test position(b, 2) == 2
    @test position(b) == [1,2]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf)]

    b = boxes1[1]
    @test position(b, 1) == 2
    @test position(b, 2) == 1
    @test position(b) == [2,1]
    @test boxbounds(b, 1) == (1.5,Inf)
    @test boxbounds(b, 2) == (-Inf,1.5)
    @test boxbounds(b) == [(1.5,Inf), (-Inf,1.5)]

    # 4-d
    world = World([0,-Inf,-5,-Inf], [Inf,Inf,50,20], fill((1,2), 4), 1)
    root = Box{2}(world)
    @test length(root) == 1
    @test length(leaves(root)) == 1
    boxes1 = Box(root, (1,2), (2,2), (2, 3, 4)) # split along dims 1 & 2
    @test length(root) == 5
    @test length(leaves(root)) == 4

    b = getleaf(root)
    @test position(b, 1) == position(b, 2) == position(b, 3) == position(b, 4) == 1
    @test position(b) == [1,1,1,1]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (-Inf,1.5)
    @test boxbounds(b, 3) == (-5,50)
    @test boxbounds(b, 4) == (-Inf,20)
    @test boxbounds(b) == [(0,1.5), (-Inf,1.5), (-5,50), (-Inf,20)]
    @test meta(b) == 1

    b = boxes1[1]
    @test position(b, 1) == 2
    @test position(b, 2) == position(b, 3) == position(b, 4) == 1
    @test position(b) == [2,1,1,1]
    @test boxbounds(b, 1) == (1.5,Inf)
    @test boxbounds(b, 2) == (-Inf,1.5)
    @test boxbounds(b, 3) == (-5,50)
    @test boxbounds(b, 4) == (-Inf,20)
    @test boxbounds(b) == [(1.5,Inf), (-Inf,1.5), (-5,50), (-Inf,20)]
    @test meta(b) == 2

    b = boxes1[2]
    @test position(b, 1) == position(b, 3) == position(b, 4) == 1
    @test position(b, 2) == 2
    @test position(b) == [1,2,1,1]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (-5,50)
    @test boxbounds(b, 4) == (-Inf,20)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf), (-5,50), (-Inf,20)]
    @test meta(b) == 3

    b = boxes1[3]
    @test position(b, 1) == 2
    @test position(b, 2) == 2
    @test position(b) == [2,2,1,1]
    @test boxbounds(b, 1) == (1.5,Inf)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (-5,50)
    @test boxbounds(b, 4) == (-Inf,20)
    @test boxbounds(b) == [(1.5,Inf), (1.5,Inf), (-5,50), (-Inf,20)]
    @test meta(b) == 4


    boxes2 = Box(boxes1[2], (3,4), (2,3), (5, 6, 7)) # split box@[1,2,1,1] along dims 3 & 4
    @test length(root) == 9
    @test length(leaves(root)) == 7
    b = getleaf(boxes1[2])
    @test position(b) == [1,2,1,1]
    @test boxbounds(b, 3) == (-5,1.5)
    @test boxbounds(b, 4) == (-Inf,2)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf), (-5,1.5), (-Inf,2)]
    @test meta(b) == 3

    b = boxes2[1]
    @test position(b, 2) == position(b, 3) == 2
    @test position(b, 1) == position(b, 4) == 1
    @test position(b) == [1,2,2,1]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (1.5,50)
    @test boxbounds(b, 4) == (-Inf,2)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf), (1.5,50), (-Inf,2)]
    @test meta(b) == 5

    b = boxes2[2]
    @test position(b, 1) == position(b, 3) == 1
    @test position(b, 2) == 2
    @test position(b, 4) == 3
    @test position(b) == [1,2,1,3]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (-5,1.5)
    @test boxbounds(b, 4) == (2,20)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf), (-5,1.5), (2,20)]
    @test meta(b) == 6

    b = boxes2[3]
    @test position(b, 2) == position(b, 3) == 2
    @test position(b, 1) == 1
    @test position(b, 4) == 3
    @test position(b) == [1,2,2,3]
    @test boxbounds(b, 1) == (0,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (1.5,50)
    @test boxbounds(b, 4) == (2,20)
    @test boxbounds(b) == [(0,1.5), (1.5,Inf), (1.5,50), (2,20)]
    @test meta(b) == 7


    boxes3 = Box(boxes2[3], (3,1), (1.8,1.3), (8, 9, 10)) # split box@[1,2,2,3] along dims 3 & 1
    @test length(root) == 13
    @test length(leaves(root)) == 10
    b = getleaf(boxes2[3])
    @test position(b) == [1,2,2,3]
    @test boxbounds(b, 1) == (0,1.15)
    @test boxbounds(b, 3) == (1.9,50)
    @test boxbounds(b) == [(0,1.15), (1.5,Inf), (1.9,50), (2,20)]
    @test meta(b) == 7

    b = boxes3[1]
    @test position(b, 1) == 1
    @test position(b, 2) == 2
    @test position(b, 3) == 1.8
    @test position(b, 4) == 3
    @test position(b) == [1,2,1.8,3]
    @test boxbounds(b, 1) == (0,1.15)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (1.5,1.9)
    @test boxbounds(b, 4) == (2,20)
    @test boxbounds(b) == [(0,1.15), (1.5,Inf), (1.5,1.9), (2,20)]
    @test meta(b) == 8

    b = boxes3[2]
    @test position(b, 1) == 1.3
    @test position(b, 2) == position(b, 3) == 2
    @test position(b, 4) == 3
    @test position(b) == [1.3,2,2,3]
    @test boxbounds(b, 1) == (1.15,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (1.9,50)
    @test boxbounds(b, 4) == (2,20)
    @test boxbounds(b) == [(1.15,1.5), (1.5,Inf), (1.9,50), (2,20)]
    @test meta(b) == 9

    b = boxes3[3]
    @test position(b, 1) == 1.3
    @test position(b, 2) == 2
    @test position(b, 3) == 1.8
    @test position(b, 4) == 3
    @test position(b) == [1.3,2,1.8,3]
    @test boxbounds(b, 1) == (1.15,1.5)
    @test boxbounds(b, 2) == (1.5,Inf)
    @test boxbounds(b, 3) == (1.5,1.9)
    @test boxbounds(b, 4) == (2,20)
    @test boxbounds(b) == [(1.15,1.5), (1.5,Inf), (1.5,1.9), (2,20)]
    @test meta(b) == 10

    # 5-d
    geom = Dict()
    root = generate_randboxes(Box{2}, 5, 100, (args...)->record_geometry!(geom, args...))
    for leaf in leaves(root)
        pos, bbs = geom[leaf]
        @test position(leaf) == pos
        @test boxbounds(leaf) == bbs
    end
end

@testset "Split iteration" begin
    f(x) = rand()

    n = 3
    world = World(fill(-Inf, n), fill(Inf, n), fill((0,1), n), rand())
    root = Box{2}(world)
    x = ones(n)
    box = addpoint!(root, x, [(1,2), (3,1)], f)
    x = [0.6, 2, 2]
    box = addpoint!(root, x, [(1,2), (3,2)], f)
    s = [split.dims for split in splits(box)]
    @test s == [(3,2), (1,2), (3,1), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(3,2), (1,2), (3,1), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(1,2), (3,2), (3,1), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(3,1), (1,2), (3,2), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(1,2), (3,1), (1,2), (3,2)]

    n = 6
    world = World(fill(-Inf, n), fill(Inf, n), fill((0,1), n), rand())
    root = Box{2}(world)
    x = ones(n); addpoint!(root, x, [(1,2), (3,4), (5,6)], f)
    x = [0.8; 0.8; 0.8; 0.2; 0.2; 0.2]; addpoint!(root, x, [(1,3), (2,5), (4,6)], f)
    x = fill(0.2, n); addpoint!(root, x, [(1,5), (3,6), (2,4)], f)
    box = find_leaf_at(root, [0.2,0,0.2,0,0.2,0])
    s = [split.dims for split in splits(box)]
    @test s == [(3,6), (2,4), (1,5), (1,2), (3,4), (1,3), (2,5), (4,6), (5,6)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(3,6), (2,4), (1,5), (1,2), (3,4), (1,3), (2,5), (4,6), (5,6)]
    box = find_leaf_at(root, [0.8,0.8,0.8,0.2,0.2,0.2])
    s = [split.dims for split in splits(box)]
    @test s == [(4,6), (2,5), (1,3), (3,4), (5,6), (1,2), (1,5), (3,6), (2,4)]
    box = find_leaf_at(root, ones(6))
    s = [split.dims for split in splits(box)]
    @test s == [(5,6), (3,4), (1,3), (2,5), (4,6), (1,2), (1,5), (3,6), (2,4)]
    box = find_leaf_at(root, [1,1,0,0,0,0])
    s = [split.dims for split in splits(box)]
    @test s == [(3,4), (1,3), (2,5), (4,6), (5,6), (1,2), (1,5), (3,6), (2,4)]
    s = [split.dims for split in splits(root)]
    @test s == [(1,2), (1,5), (3,6), (2,4), (3,4), (1,3), (2,5), (4,6), (5,6)]

    n = 8
    world = World(fill(-Inf, n), fill(Inf, n), fill((0,1), n), rand())
    for i = 1:20
        root = Box{2}(world)
        x = randn(n); addpoint!(root, x, [(1,2), (3,4), (5,6), (7,8)], f)
        x = randn(n); addpoint!(root, x, [(3,7), (4,8), (1,5), (2,6)], f)
        x = randn(n); addpoint!(root, x, [(6,8), (1,3), (2,4), (5,7)], f)
        x = randn(n); addpoint!(root, x, [(4,5), (1,6), (2,7), (3,8)], f)
        box = minimum(root)
        s = [split.dims for split in splits(box)]
        @test length(s) == 16 && length(unique(s)) == 16
    end
end

@testset "Collinearity" begin
    for i = 1:20
        root = generate_randboxes(Box{1}, 5, 20)
        X = collect_positions(root)
        for leaf in leaves(root)
            @test CoordinateSplittingPTrees.ncollinear(leaf) == collinear_dims(X, position(leaf))
        end
    end

    for i = 1:20
        root = generate_randboxes(Box{2}, 5, 20)
        X = collect_positions(root)
        for leaf in leaves(root)
            @test CoordinateSplittingPTrees.ncollinear(leaf) == collinear_dims(X, position(leaf))
        end
    end

end

@testset "Display" begin
    io = IOBuffer()
    io2 = IOBuffer()
    # CS1, 1-d
    world = World([0], [1], [(0,1)], 10)
    root = Box{1}(world)
    box = Box(root, 1, 1, 20)[1]
    box1 = Box(box, 1, 0.5, 30)[1]
    splitprint(io, root)
    @test String(take!(io)) == "(1,)[l, (1,)[l, l]]"
    splitprint_colored(io, root, box1)
    print_with_color(:cyan, io2, "(1,)")
    print(io2, "[l, ")
    print_with_color(:cyan, io2, "(1,)")
    print(io2, "[l, ")
    print_with_color(:light_red, io2, "l")
    print(io2, "]]")
    @test String(take!(io)) == String(take!(io2))
    print_tree(io, root)
    @test String(take!(io)) == """
(1,)
├─ Box10@[0.0]
└─ (1,)
   ├─ Box20@[1.0]
   └─ Box30@[0.5]
"""

    # CS1, 2-d
    world = World([0,-Inf], [Inf,Inf], [(1,2), (1,2)], 'A')
    root = Box{1}(world)
    box1 = Box(root, 1, 2, 'B')[1]
    box2 = Box(getleaf(root), 2, 2, 'C')[1]
    print_tree(io, root)
    @test String(take!(io)) == """
(1,)
├─ (2,)
│  ├─ Box'A'@[1.0, 1.0]
│  └─ Box'C'@[1.0, 2.0]
└─ Box'B'@[2.0, 1.0]
"""

    # CS2, 3-d
    world = World(fill(-Inf,3), fill(Inf,3), fill((1,2), 3), 100)
    root = Box{2}(world)
    boxes1 = Box(root, (1,2), (10,20), (200,300,400))
    boxes2 = Box(boxes1[2], (3,1), (-10,3.5), (500,600,700))
    print_tree(io, root)
    @test String(take!(io)) == """
(1, 2)
├─ Box100@[1.0, 1.0, 1.0]
├─ Box200@[10.0, 1.0, 1.0]
├─ (3, 1)
│  ├─ Box300@[1.0, 20.0, 1.0]
│  ├─ Box500@[1.0, 20.0, -10.0]
│  ├─ Box600@[3.5, 20.0, 1.0]
│  └─ Box700@[3.5, 20.0, -10.0]
└─ Box400@[10.0, 20.0, 1.0]
"""
end
