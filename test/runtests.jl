using CoordinateSplittingPTrees
import AbstractTrees
using Test, Random, LinearAlgebra, SparseArrays
using GLPK
include("functions.jl")

struct DummyMeta
    val::Float64
end

@testset "World" begin
    world = @inferred(World([0], [1], [0], 10))
    @test eltype(world.lower) == Float64
    world = @inferred(World([0.0], [1], [0], 10))
    @test eltype(world.lower) == Float64
    world = @inferred(World(zeros(5), ones(5), rand(5), 10))
    @test eltype(world.lower) == Float64
    @test_throws ArgumentError World(zeros(2), ones(2), [7.0,0.5], 0)
    world = @inferred(World(zeros(2), [Inf,1], [7.0,0.5], 0))
    @test CoordinateSplittingPTrees.baseposition(world) == [7.0, 0.5]
    world = @inferred(World([1,2], 0))
    @test world.lower == [-Inf,-Inf]
    @test world.upper == [Inf,Inf]
    @test CoordinateSplittingPTrees.baseposition(world) == [1.0,2.0]
    world = @inferred(World([1,2], x->3.0))
    @test world.meta == 3.0

    # Behavior reserved for use by QuadSplit (2-tuple splits)
    world = @inferred(World(zeros(3), fill(3, 3), fill((1,2), 3), DummyMeta(0)))
    @test eltype(world.lower) == Float64
    @test CoordinateSplittingPTrees.baseposition(world) == ones(3)
    world = World(zeros(3), fill(3, 3), fill((2,1), 3), DummyMeta(0))
    @test CoordinateSplittingPTrees.baseposition(world) == fill(2, 3)
    @test_throws ArgumentError World(zeros(3), fill(3, 3), fill((1,1), 3), DummyMeta(0))
    world = World(zeros(3), fill(3, 3), fill((2,1), 3), DummyMeta(0))
    @test CoordinateSplittingPTrees.baseposition(world) == fill(2, 3)
    world = World(fill((2,1), 3), DummyMeta(0))
    @test world.lower == [-Inf,-Inf,-Inf]
    @test world.upper == [Inf,Inf,Inf]
    @test CoordinateSplittingPTrees.baseposition(world) == fill(2, 3)

    world = World(fill(-Inf, 3), fill(Inf, 3), fill((1,2.5), 3), 0)
    root = Box{2}(world)
    @test position(root) == ones(3)
    box = CoordinateSplittingPTrees.addpoint_distinct!(root, [5,1.0,3], x->0)
    @test position(box) == [5,2.5,3]
end

@testset "Geometry and iteration, CS1" begin
    # For comparing boxbounds
    myapproxeq(t1::Tuple{Real,Real}, t2::Tuple{Real,Real}) = t1[1] ≈ t2[1] && t1[2] ≈ t2[2]
    myapproxeq(v1::Vector, v2::Vector) = all(myapproxeq.(v1, v2))

    # 1-d
    world = World([0], [1], [0], 10)
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
    @test CoordinateSplittingPTrees.epswidth(boxbounds(box, 1)) == eps()
    @test CoordinateSplittingPTrees.epswidth(boxbounds(getleaf(root), 1)) == eps(0.5)
    # evaluation point is at the boundary of parents
    @test_throws AssertionError Box(root, 1, 0.5, 30)  # can only split leaves
    @test_throws ErrorException Box(getleaf(root), 1, 0.5, 30)  # can't eval at upper edge
    boxes = collect(root)
    @test boxes == [root, root.split.self, box]
    boxes = collect(box)
    @test boxes == [box]
    boxes = collect(leaves(root))
    @test boxes == [root.split.self, box]
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
    @test position(leaf) ∈ root
    @test position(leaf) ∈ leaf

    world = World([0], [Inf], [0], 10) # with infinite size
    root = Box{1}(world)
    box = Box(root, 1, 1, 20)[1]
    @test boxbounds(box) == [(0.5,Inf)]
    @test boxbounds(box, 1) == (0.5,Inf)
    @test CoordinateSplittingPTrees.epswidth(boxbounds(box, 1)) == eps(0.5)
    @test CoordinateSplittingPTrees.epswidth(boxbounds(getleaf(root), 1)) == eps(0.5)

    world = World([0], [1], [1], 10)
    root = Box{1}(world)
    box1 = Box(root, 1, 0, 20)[1]
    leaf = find_leaf_at(root, 0.5)
    @test leaf == getleaf(root)

    geom = Dict()
    root = generate_randboxes(Box{1}, 1, 10, (args...)->record_geometry!(geom, args...))
    for leaf in leaves(root)
        pos, bbs = geom[leaf]
        @test position(leaf) == pos
        @test boxbounds(leaf) == bbs
    end

    # 2-d
    world = World([0,-Inf], [Inf,Inf], [1,1], nothing)
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

    @test all(CoordinateSplittingPTrees.splittable, boxbounds(b1_1))

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

    # 1-d
    world = World([-Inf], [Inf], [1], 0)
    root = Box{2}(world)
    addpoint!(root, [1.2], x->0)
    # test for displacement along fictive dimension
    chldrn = collect(AbstractTrees.children(root))
    @test !CoordinateSplittingPTrees.isfake(chldrn[1])
    @test !CoordinateSplittingPTrees.isfake(chldrn[2])
    @test  CoordinateSplittingPTrees.isfake(chldrn[3])
    @test  CoordinateSplittingPTrees.isfake(chldrn[4])
    @test length(leaves(root)) == 2
    box1 = addpoint!(root, [0.75], x->0)
    @test length(leaves(root)) == 3
    box2 = addpoint!(root, [0.5], x->0)
    @test length(leaves(root)) == 4
    @test boxbounds(box2, 1) == (-Inf, 5/8)
    @test boxbounds(box1, 1) == (-Inf, 7/8)
    @test boxbounds(getleaf(box1), 1) == (5/8, 7/8)


    world = World([0], [1], [0], 10)
    root = Box{1}(world)
    box1 = Box(root, 1, 1, 20)[1]
    leaf = find_leaf_at(root, 0.5)
    @test leaf == box1
    world = World([0], [1], [1], 10)
    root = Box{1}(world)
    box1 = Box(root, 1, 0, 20)[1]
    leaf = find_leaf_at(root, 0.5)
    @test leaf == getleaf(root)

    # 2-d
    world = World([0,-Inf], [Inf,Inf], [1,1], nothing)
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

    # 3-d (fake dimensions and iteration)
    world = World(fill(-Inf,3), fill(Inf,3), fill(0,3), 1)
    root = Box{2}(world)
    @test length(root) == 1
    @test length(leaves(root)) == 1
    boxes1 = Box(root, (1,2), (1,1), (2, 3, 4)) # split along dims 1 & 2
    @test length(root) == 5
    @test length(leaves(root)) == 4
    boxes2 = Box(boxes1[end], (3,4), (1,1), (5, 6, 7)) # split along dims 1 & 2
    @test length(leaves(root)) == 5
    v = [value(leaf) for leaf in leaves(root)]
    @test v == [1:5;]

    # 4-d
    world = World([0,-Inf,-5,-Inf], [Inf,Inf,50,20], fill(1, 4), 1)
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

    # Odd dimensionality for CSp with p even (fictive dimensions)
    n = 3
    world = World(fill(-Inf, n), fill(Inf, n), fill(0, n), rand())
    root = Box{2}(world)
    x = ones(n)
    box = addpoint!(root, x, [(1,2), (3,)], f)
    @test position(box) == x
    x = [0.6, 2, 2]
    box = addpoint!(root, x, [(2,3), (1,)], f)
    @test position(box) == [0.6,2,2]
    @test boxbounds(box) == [(0.5, 0.8), (1.5,Inf), (1.5,Inf)]

    s = [split.dims for split in splits(box)]
    @test s == [(1,4), (2,3), (3,4), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(1,4), (2,3), (3,4), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(2,3), (1,4), (3,4), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(3,4), (2,3), (1,4), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(1,2), (3,4), (2,3), (1,4)]

    n = 6
    world = World(fill(-Inf, n), fill(Inf, n), fill(0, n), rand())
    root = Box{2}(world)
    x = ones(n); box = addpoint!(root, x, [(1,2), (3,4), (5,6)], f)
    position(box) == x
    x = [0.8; 0.8; 0.8; 0.2; 0.2; 0.2]; addpoint!(root, x, [(1,3), (2,5), (4,6)], f)
    x = fill(0.2, n); addpoint!(root, x, [(1,5), (3,6), (2,4)], f)
    box = find_leaf_at(root, [0.2,0,0.2,0,0.2,0])
    s = [split.dims for split in splits(box)]
    @test s == [(2,4), (3,6), (1,5), (3,4), (1,3), (2,5), (4,6), (5,6), (1,2)]
    box = box.parent
    s = [split.dims for split in splits(box)]
    @test s == [(3,6), (2,4), (1,5), (3,4), (1,3), (2,5), (4,6), (5,6), (1,2)]
    box = find_leaf_at(root, [0.8,0.8,0.8,0.2,0.2,0.2])
    s = [split.dims for split in splits(box)]
    @test s == [(4,6), (2,5), (1,3), (5,6), (3,4), (1,5), (3,6), (2,4), (1,2)]
    box = find_leaf_at(root, ones(6))
    s = [split.dims for split in splits(box)]
    @test s == [(5,6), (1,3), (2,5), (4,6), (3,4), (1,5), (3,6), (2,4), (1,2)]
    box = find_leaf_at(root, [1,1,0,0,0,0])
    s = [split.dims for split in splits(box)]
    @test s == [(1,3), (2,5), (4,6), (5,6), (3,4), (1,5), (3,6), (2,4), (1,2)]
    s = [split.dims for split in splits(root)]
    @test s == [(1,2), (1,5), (3,6), (2,4), (3,4), (1,3), (2,5), (4,6), (5,6)]

    n = 8
    world = World(fill(-Inf, n), fill(Inf, n), fill(0, n), rand())
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

@testset "Chains" begin
    # Cases with and without fictive dimensions
    f(x) = rand()
    for (n, dimlistss) in ((3, ([(1,2), (3,)],  [(3,1), (2,)])),
                           (4, ([(1,2), (3,4)], [(3,1), (2,4)])),
                           (4, ([(1,2), (3,4)], [(1,2), (3,4)])))
        x0 = randn(n)
        world = World(fill(-Inf, n), fill(Inf, n), x0, f(x0))
        root = box = Box{2}(world)
        chaintops = typeof(root)[]
        repeats = dimlistss[2] == dimlistss[1]
        for dimlists in dimlistss
            x = position(box) .+ 1
            bx = box
            box = addpoint!(box, x, dimlists, f)
            for i = 1:ceil(Int, n/2)
                push!(chaintops, bx)
                if repeats
                    # If the dimlists repeat, then any split box is
                    # top of some chain
                    bx = bx.split.others.children[end]
                end
            end
        end
        if repeats
            chaintops[end] = chaintops[end].parent # need full chain
        end
        i = length(chaintops)
        while !isroot(box)
            top, success = chaintop(box)
            @test success
            @test top == chaintops[i]
            ntests = 0
            for (j, split) in enumerate(chain(top))
                @test j < 3
                if j == 1
                    @test split == top.split
                    ntests += 1
                elseif j == 2
                    @test split == top.split.others.children[end].split
                    ntests += 1
                end
            end
            @test ntests == 2
            for child in box.parent.split.others.children
                top, success = CoordinateSplittingPTrees.chaintop(child)
                @test success
                @test top == chaintops[i]
            end
            i -= 1
            box = box.parent
        end
    end
end

@testset "p-th order direct coefficients" begin
    bprod(x, B::AbstractVector) = B'*x
    bprod(x, B::AbstractMatrix) = x'*B*x/2
    function bprod(x, B::AbstractArray{T,3}) where T
        s = zero(T)
        for I in CartesianIndices(size(B))
            p = oneunit(T)
            for d in I.I
                p *= x[d]
            end
            s += p*B[I]
        end
        return s/6
    end

    for p in (1, 2, 3)
        for n in (6, 7)
            x0 = zeros(n)
            world = World(fill(-Inf, n), fill(Inf, n), x0, 0.0)
            sz = ntuple(d->n, p)
            B = CoordinateSplittingPTrees.SymmetricArray(randn(sz))
            f(x) = bprod(x, B)
            root = Box{p}(world)
            filled = CoordinateSplittingPTrees.SymmetricArray(falses(sz))
            if p > 1
                for I in CartesianIndices(sz)
                    if anydups(I.I)
                        filled[I] = true
                    end
                end
            end
            # Generate splits
            while !all(filled)
                local pairings
                while true
                    pairings = dimpair(randperm(n), p)
                    isnovel = true
                    for pr in pairings
                        length(pr) == p || continue
                        isnovel = !filled[pr...]
                    end
                    isnovel && break
                end
                for pr in pairings
                    length(pr) == p || break
                    filled[pr...] = true
                end
                addpoint!(root, randn(n), pairings, f)
            end
            for box in leaves(root)
                Cp = CoordinateSplittingPTrees.coefficients_p(box)
                testequal_offdiag(Cp, B)
            end
        end
    end
end

hesscount = 0

@testset "Sparse Hessians" begin
    function fsparse(x)
        dx = diff(x)       # only (i,i+1) Q terms are nonzero
        return sum(abs2, dx)/2
    end
    x0 = [0.1,0.2,0.3,0.4]
    world = World(x0, fsparse)
    root = Box{2}(world)
    box = addpoint!(root, 2*x0, [(1,3), (2,4)], fsparse) # neither split has nonzero in Hessian
    box = addpoint!(root, 3*x0, [(2,3), (1,4)], fsparse)
    box = addpoint!(root, 4*x0, [(1,4), (2,3)], fsparse)
    box = addpoint!(root, 5*x0, [(1,2), (3,4)], fsparse)
    callback = (box,val)->(global hesscount; hesscount+=1)
    Q = zeros(4, 4)
    @test_throws MethodError CoordinateSplittingPTrees.coefficients_p!(Q, box, callback)  # type must enforce symmetry
    Q = CoordinateSplittingPTrees.SymmetricArray(Q)
    CoordinateSplittingPTrees.coefficients_p!(Q, box, callback)
    for i = 1:3
        @test Q[i,i+1] ≈ -1
        @test Q[i+1,i] ≈ -1
        @test isnan(Q[i,i])
    end
    @test abs((Q[1,3])) < 1e-8
    @test abs((Q[2,4])) < 1e-8
    @test abs((Q[1,4])) < 1e-8
    global hesscount
    @test hesscount == 24
    Q = SymTridiagonal(zeros(4), zeros(3))
    hesscount = 0
    CoordinateSplittingPTrees.coefficients_p!(Q, box, callback)
    for i = 1:3
        @test Q[i,i+1] ≈ -1
        @test Q[i+1,i] ≈ -1
        @test isnan(Q[i,i])
    end
    @test hesscount == 12
    Q = sparse([1,1,2,2,3,3,4,1], [1,2,2,3,3,4,4,4], zeros(8))
    @test_throws MethodError CoordinateSplittingPTrees.coefficients_p!(Q, box, callback)
    Q = CoordinateSplittingPTrees.SymmetricArray(Q)
    @test CoordinateSplittingPTrees.nnz_offdiag_sym(Q) == 4
    hesscount = 0
    CoordinateSplittingPTrees.coefficients_p!(Q, box, callback)
    for i = 1:3
        @test Q[i,i+1] ≈ -1
        @test Q[i+1,i] ≈ -1
        @test isnan(Q[i,i])
    end
    @test hesscount == 16
end

@testset "Choosing split dimensions" begin
    f(x) = rand()
    n = 2
    x0 = randn(n)
    world = World(fill(-Inf, n), fill(Inf, n), x0, f(x0))
    root = Box{2}(world)
    pairs = CoordinateSplittingPTrees.choose_dimensions(root)
    @test pairs == [(1,2)]
    box = addpoint!(root, randn(n), pairs, f)
    pairs = CoordinateSplittingPTrees.choose_dimensions(box)
    @test pairs == [(1,2)]

    n = 3
    x0 = randn(n)
    world = World(fill(-Inf, n), fill(Inf, n), x0, f(x0))
    pairseq = []
    root = Box{2}(world)
    pairs = CoordinateSplittingPTrees.choose_dimensions(root)
    @test pairs == [(1,2), (3,)]
    push!(pairseq, pairs)
    box = addpoint!(root, randn(n), pairs, f)
    pairs = CoordinateSplittingPTrees.choose_dimensions(box)
    push!(pairseq, pairs)
    @test pairs[1] ∈ ((1,3), (2,3))
    box = addpoint!(root, randpoint_inside(box), pairs, f)
    pairs2 = CoordinateSplittingPTrees.choose_dimensions(box)
    push!(pairseq, pairs2)
    @test pairs2[1] ∈ ((1,3), (2,3))
    @test pairs2[1] != pairs[1]
    box = addpoint!(root, randpoint_inside(box), pairs2, f)
    for i = 1:3
        pairs = CoordinateSplittingPTrees.choose_dimensions(box)
        @test pairs == pairseq[i]
        box = addpoint!(root, randpoint_inside(box), pairs, f)
    end

    root = Box{2}(world)
    pairs = CoordinateSplittingPTrees.choose_dimensions(root)
    box = addpoint!(root, randn(n), pairs, f)
    pairs = CoordinateSplittingPTrees.choose_dimensions(box)
    boxd = addpoint!(root, randpoint_inside(box), pairs, f)
    boxp = box.parent.split.others.children[2]
    pairs2 = CoordinateSplittingPTrees.choose_dimensions(boxp)
    @test pairs2[1] ∈ ((1,3), (2,3))
    @test pairs2[1] != pairs[1]
    boxpd = addpoint!(root, randpoint_inside(boxp), pairs2, f)
    @test CoordinateSplittingPTrees.choose_dimensions(boxpd) ==
          CoordinateSplittingPTrees.choose_dimensions(boxd)  == [(1,2), (3,)]
end

@testset "Qcoef" begin
    using CoordinateSplittingPTrees: Qcoef_value, Qcoef_lineq, chi
    for n in (7, 8)
        dims = randperm(n)
        seen = falses(n)
        prec = falses(n, n)
        for i = 1:2:n
            d1 = d2 = dims[i]
            prec[:,d1] = seen
            if i < n
                d2 = dims[i+1]
                prec[:,d2] = seen
            end
            seen[d1] = true
            if i < n
                seen[d2] = true
            end
        end
        b = randn(n)
        y = randn(n)
        for i = 1:n  # ensure y is distinct from b in all coordinates
            while y[i] == b[i]
                y[i] = randn()
            end
        end
        x = copy(b)
        for i = 1:2:n
            d1 = d2 = dims[i]
            Δx1 = Δx2 = y[d1] - b[d1]
            if i < n
                d2 = dims[i+1]
                Δx2 = y[d2] - b[d2]
            end
            for k = 1:n
                (k == d1 || k == d2) && continue
                @test Qcoef_lineq(d1, k, Δx1, x[k], b[k], y[k], prec[d1,k]) == 0
                @test Qcoef_lineq(d2, k, Δx2, x[k], b[k], y[k], prec[d2,k]) == 0
            end
            x[d1] = y[d1]
            if i < n
                x[d2] = y[d2]
            end
            for k = 1:n
                (k == d1 || k == d2) && continue
                @test Qcoef_value(k, d1, x, b, y, prec) == 0
                @test Qcoef_value(d1, k, x, b, y, prec) == 0
                @test Qcoef_value(k, d2, x, b, y, prec) == 0
                @test Qcoef_value(d2, k, x, b, y, prec) == 0
                @test abs((x[d1] - b[d1]) * (x[k] - chi(k, d1, b, y, prec))/2 +
                          (x[k] - b[k]) * (x[d1] - chi(d1, k, b, y, prec))/2) < 1e-12
                @test abs((x[d2] - b[d2]) * (x[k] - chi(k, d2, b, y, prec))/2 +
                          (x[k] - b[k]) * (x[d2] - chi(d2, k, b, y, prec))/2) < 1e-12
            end
            @test Qcoef_value(d1, d1, x, b, y, prec) == 0
            @test Qcoef_value(d2, d2, x, b, y, prec) == 0
            @test abs((x[d1] - b[d1]) * (x[d1] - chi(d1, d1, b, y, prec))/2 +
                      (x[d1] - b[d1]) * (x[d1] - chi(d1, d1, b, y, prec))/2) < 1e-12
            @test abs((x[d2] - b[d2]) * (x[d2] - chi(d2, d2, b, y, prec))/2 +
                      (x[d2] - b[d2]) * (x[d2] - chi(d2, d2, b, y, prec))/2) < 1e-12
            @test Qcoef_value(d1, d2, x, b, y, prec) ≈ (
                (x[d1] - b[d1]) * (x[d2] - chi(d2, d1, b, y, prec))/4 +
                (x[d2] - b[d2]) * (x[d1] - chi(d1, d2, b, y, prec))/4)
        end
        x = randn(n)
        for j = 1:n, i = 1:n
            @test Qcoef_value(i, j, x, b, y, prec) ≈ (
                (x[i] - b[i]) * (x[j] - chi(j, i, b, y, prec))/4 +
                (x[j] - b[j]) * (x[i] - chi(i, j, b, y, prec))/4)
        end
    end
end

@testset "CS2 full models" begin
    for n = (2, 3, 7, 8)
        B = CoordinateSplittingPTrees.SymmetricArray(randn(n, n))
        f(x) = (x'*B*x)/2
        x0 = randn(n)
        world = World(fill(-Inf, n), fill(Inf, n), x0, f(x0))
        for iter = 1:10  # ensure a variety of splitting patterns
            root = box = Box{2}(world)
            while true
                box = addpoint!(root, randn(n), f)
                Cp = CoordinateSplittingPTrees.coefficients_p(box)
                !hasnan_offdiag(Cp) && break
            end
            # One extra to make sure we can determine the diagonal components
            box = addpoint!(root, randn(n), f)
            c, g, Q, b, y, prec, firstmatch = CoordinateSplittingPTrees.fit_quadratic_native(box)
            for leaf in leaves(root)
                x = position(leaf)
                @test modelvalue(c, g, Q, b, y, prec, x) ≈ f(x) rtol=1e-7
                @test modelvalue_chi(c, g, Q, b, y, prec, x) ≈ f(x) rtol=1e-7
            end
            c, g, Q, b = CoordinateSplittingPTrees.fit_quadratic(box)
            @test Q ≈ B
            @test g ≈ Q*b
            x = position(box)
            @test c + g'*(x - b) + (x-b)'*Q*(x-b)/2 ≈ f(x) rtol=1e-8
            @test abs(c - g'*b + (b'*Q*b)/2) < 1e-8
            # gdiag approach (which can yield different answers for non-quadratic functions,
            # but should be identical for quadratic)
            Q = CoordinateSplittingPTrees.coefficients_p(box)
            c, g, Q, b = CoordinateSplittingPTrees.fit_quadratic_gdiag!(Q, box)
            @test Q ≈ B
            @test g ≈ Q*b
        end
    end
end

@testset "Lowerbound models" begin
    f(x) = sum(abs2, x)/2 + x[1]*x[2]/5
    world = World([1.0,1.0,1.0,1.0], f)
    n = ndims(world)
    root = Box{2}(world)
    for i = 1:n+2
        addpoint!(root, randn(n), f)
    end
    box = minimum(root)

    updater, model, gp, dp, Q = CoordinateSplittingPTrees.lowerbound_model(box, GLPK.Optimizer())
    c, g, Q, b = CoordinateSplittingPTrees.fit_quadratic_lowerbound!(Q, updater, model, gp, dp, box)
    Qtarget = Matrix(Diagonal(ones(n)))
    Qtarget[1,2] = Qtarget[2,1] = 0.2
    @test Q ≈ Qtarget rtol=0.01
    @test g ≈ Q*b rtol=0.01

    # Check that it works with an excess of points
    updater, model, gp, dp, Q = CoordinateSplittingPTrees.lowerbound_model(box, GLPK.Optimizer(), 3n)
    c, g, Q, b = CoordinateSplittingPTrees.fit_quadratic_lowerbound!(Q, updater, model, gp, dp, box)
    Qtarget = Matrix(Diagonal(ones(n)))
    Qtarget[1,2] = Qtarget[2,1] = 0.2
    @test Q ≈ Qtarget rtol=0.01
    @test g ≈ Q*b rtol=0.01

    # Check that nothing breaks if we ask for more constraints than we have boxes
    updater, model, gp, dp, Q = CoordinateSplittingPTrees.lowerbound_model(box, GLPK.Optimizer(), length(leaves(root))+1)
    c, g, Q, b = CoordinateSplittingPTrees.fit_quadratic_lowerbound!(Q, updater, model, gp, dp, box)
    Qtarget = Matrix(Diagonal(ones(n)))
    Qtarget[1,2] = Qtarget[2,1] = 0.2
    @test Q ≈ Qtarget rtol=0.01
    @test g ≈ Q*b rtol=0.01
end

@testset "Gauss elim" begin
    ige = CoordinateSplittingPTrees.IGE{Float64}(3)
    insert!(ige, [0,1,0], 1)
    insert!(ige, [1,0,0], 2)
    insert!(ige, [0,0,1], 3)
    x = CoordinateSplittingPTrees.solve(ige)
    @test x ≈ [2,1,3]
    empty!(ige)
    U, _ = qr(randn(3,3))  # random unitary matrix
    A = U*Diagonal(range(1, stop=3, length=3))*U'  # random posdef with eigvals 1,2,3
    rhs = randn(3)
    xtrue = A\rhs
    for i = 1:3
        insert!(ige, A[i,:], rhs[i])
        if i < 3
            @test !iscomplete(ige)
            @test_throws LinearAlgebra.LAPACKException CoordinateSplittingPTrees.solve(ige)
        end
    end
    @test iscomplete(ige)
    x = CoordinateSplittingPTrees.solve(ige)
    @test x ≈ xtrue
    insert!(ige, A[3,:], 0)
    x = CoordinateSplittingPTrees.solve(ige)
    @test x ≈ xtrue
    A = U*Diagonal([1.0,1.0,1e-12])*U'  # random posdef one small eigval
    empty!(ige)
    for i = 1:3
        insert!(ige, A[i,:], rhs[i])
    end
    x = CoordinateSplittingPTrees.solve(ige)
    @test x ≈ A \ rhs rtol=1e-4
end

@testset "posdef" begin
    Q = Float64[4 1; 1 4]
    Qp = CoordinateSplittingPTrees.possemidef(Q)
    @test Qp == Q
    Q = Float64[1 4; 4 1]
    Qp = CoordinateSplittingPTrees.possemidef(Q)
    @test Qp == [4 4; 4 4]
    Q = Float64[4.5 3; 3 0.5]
    Qp = CoordinateSplittingPTrees.possemidef(Q)
    @test Qp == [9 3; 3 1]
    # This case is deliberately weaker in what it tests, since the
    # desired behavior for 0 diagonal entries is less clear.
    Q = Float64[0 2; 2 1]
    Qp = CoordinateSplittingPTrees.possemidef(Q)
    @test all(isfinite, Qp) && Qp[1,2] == Q[1,2]
    cholesky(Qp) # test only that it's positive-definite
end

@testset "Display" begin
    io = IOBuffer()
    io2 = IOBuffer()
    # CS1, 1-d
    world = World([0], [1], [0], 10)
    root = Box{1}(world)
    box = Box(root, 1, 1, 20)[1]
    box1 = Box(box, 1, 0.5, 30)[1]
    splitprint(io, root)
    @test String(take!(io)) == "(1,)[l, (1,)[l, l]]"
    splitprint_colored(io, root, box1)
    printstyled(io2, "(1,)", color=:cyan)
    print(io2, "[l, ")
    printstyled(io2, "(1,)", color=:cyan)
    print(io2, "[l, ")
    printstyled(io2, "l", color=:light_red)
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
    world = World([0,-Inf], [Inf,Inf], [1,1], 'A')
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
    world = World(fill(-Inf,3), fill(Inf,3), fill(1, 3), 100)
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

@testset "Box metadata" begin
    world = World([1, 1], 10.0)
    root = Box{2}(world, (-1, 1))
    local incrementor
    let counter = 1
        incrementor(box) = (counter += 1; (-counter,counter))
    end
    addpoint!(root, [2.0, 1.8], x->sum(x), incrementor)
    @test root.metabox == (-1, 1)
    i = 1
    for leaf in leaves(root)
        i += 1
        @test leaf.metabox == (-i, i)
    end
end
