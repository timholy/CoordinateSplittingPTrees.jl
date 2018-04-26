boxcoordtype(::Type{T}) where T<:AbstractFloat = T
boxcoordtype(::Type{T}) where T<:Real = Float64

const Vec2{T<:Real} = Union{Tuple{T,T}, AbstractVector{T}}

check2(v::Vec2) = v
function check2(v::AbstractVector)
    @noinline throwdm(v) = throw(DimensionMismatch("vector must have length 2, got axis range $(Base.indices1(v))"))
    Base.indices1(v) == Base.OneTo(2) || throwdm(v)
    return v
end


# The overall domain of the problem, plus initial evaluation locations
# and the function value at the initial point. Every box within the
# world will hold a link to this structure
struct World{T<:Real,M}
    lower::Vector{T}      # lower bounds on parameters
    upper::Vector{T}      # upper bounds on parameters
    splits::Vector{Tuple{T,T}}  # initial set of positions along each dimension
    meta::M               # metadata at baseposition (see below)

    function World{T,M}(lower, upper, splits, meta) where {T,M}
        # Validate the inputs
        N = length(lower)
        length(lower) == length(upper) == length(splits) == N ||
            throw(DimensionMismatch("lower, upper, and splits must have length $N (got $(length(lower)), $(length(upper)), and $(length(splits))"))
        for s in splits
            s[1] != s[2] || throw(ArgumentError("split points must be distinct, got $s"))
        end
        return new{T,M}(lower, upper, splits, meta)
    end
end

World(lower::AbstractVector{T}, upper::AbstractVector{T}, splits::AbstractVector{MM}, meta) where {T<:Real,MM<:Vec2{T}} =
    World{boxcoordtype(T), typeof(meta)}(lower, upper, splits, meta)

"""
    world = World(lower::AbstractVector, upper::AbstractVector, splits::AbstractVector{<:Tuple{Real,Real}}, meta)

Return a structure representing the rectangular domain and
initialization of a SplitCoordinatewiseTree.  `lower` and `upper` are
the bounds of the coordinates, `splits` is a vector of coordinate
2-values,

    splits = [(x1, x2), (y1, y2), ...]

representing a set of potential locations for box placement along each
coordinate.  `meta` is the metadata at `baseposition(world)`, aka
`[x1, y1, ...]` (see [`baseposition`](@ref))).
"""
function World(lower::AbstractVector, upper::AbstractVector, splits::AbstractVector, meta)
    T = promote_type(eltype(lower), eltype(upper))
    for s in splits
        @assert(s isa Tuple{Real,Real} || (s isa AbstractVector && length(s) == 2))
        T = promote_type(T, typeof(s[1]), typeof(s[2]))
    end
    return World{boxcoordtype(T),typeof(meta)}(lower, upper, splits, meta)
end

Base.ndims(world::World) = length(world.splits)

"""
    x = baseposition(world)

Return the initial box location, defined as the first element of each
coordinate of `world.splits` (see [`World`](@ref)).
"""
baseposition(world::World{T}) where T = baseposition(T, world.splits)
baseposition(::Type{T}, splits::AbstractVector) where T = T[s[1] for s in splits]
baseposition(splits::AbstractVector) = [s[1] for s in splits]

## Box

# The tree structure was designed to be non-redundant: there is only
# one way to get any value of interest.  We could copy some data and
# reduce traversal of the tree, but at the risk of values going
# out-of-sync.
# The exception is the world structure, to which a reference is held
# by every box. Without introducing type-instability there may not be
# a way around this.

# For the types below,
# - p is the degree of splitting (the number of dimensions that we split simultaneously)
# - L = 2^p-1 is the number of new (non-self) children created when splitting a box
# - T is the type used for spatial positions
# - M is the type used for the metadata associated with each n-dimensional point
# - B is a parameter representing Box before that type exists (avoids circular definitions)

struct Children{M,B,L}
    children::NTuple{L,B} # the non-self child nodes
    metas::NTuple{L,M}    # metadata for the evaluation points corresponding to new children
end

"""
    struct Split
        dims       # a tuple like (3,5) holding the splitting dimensions
        xs         # a tuple like (1.2,-15.3), the new positions along the splitting dimensions
        self       # the child-box with the same position as the parent box
        others     # the other children and metadata of the parent box
    end
"""
struct Split{p,T,M,B,L}
    dims::NTuple{p,Int}
    xs::NTuple{p,T}
    self::B
    others::Children{M,B,L}
end
Split{p,T}(dims::NTuple{p,Integer},
           xs::NTuple{p,Any},
           self::B,
           others::Children{M,B,L}) where {T,p,M,B,L} =
   Split{p,T,M,B,L}(dims, xs, self, others)

mutable struct Box{p,T,M,L}
    world::World{T,M}        # the overall problem domain
    parent::Box{p,T,M,L}     # the node above this one
    childindex::UInt8        # which of parent's children is this? (0=self)
    split::Split{p,T,M,Box{p,T,M,L},L}  # undefined if this box is a leaf node

    function Box{p,T,M,L}(world::World{T,M}) where {p,T,M,L}
        p > 8 && error("degree must be less than or equal to 8 (change childindex::UInt8 for p>8)")
        root = new{p,T,M,L}(world)
        root.parent = root
        root.childindex = 0
        return root
    end

    function Box{p,T,M,L}(parent::Box{p,T,M,L}, splitdims::NTuple{p,Integer}, xs::NTuple{p,Real}, metas::NTuple{L,M}) where {p,T,M,L}
        function box(parent::Box{p,T,M,L}, childindex::Integer) where {p,T,M,L}
            return new{p,T,M,L}(parent.world, parent, childindex)
        end

        @noinline throw1(x, bb) = error("out-of-bounds evaluation $x (bounds are $bb)")
        @noinline throw2(x, bb, u) = error("cannot split at the upper edge $x because the boxbounds $bb are not at the upper world edge $u")
        @noinline throw3(x, sd) = error("position $x is identical to the parent along $sd")

        @assert(isleaf(parent))
        for (splitdim,x) in zip(splitdims, xs)
            bb = boxbounds(parent, splitdim)
            if (x < bb[1]) | (x > bb[2])
                throw1(x, bb)
            elseif x == bb[2] && bb[2] != (u = parent.world.upper[splitdim])
                # To ensure that each point corresponds to a unique leaf
                # (needed for proper behavior of `find_leaf_at`), we
                # essentially define the boxbounds as the half-open
                # interval [lower, upper).  Therefore we have to prevent
                # splitting with an evaluation point at the upper edge of
                # box. (This forces the user to split the other box that
                # shares this edge.) The only exception is for boxes that
                # extend to the upper edge of the world bounds, for which
                # the bounds are defined as the closed interval [lower, upper].
                throw2(x, bb, u)
            end
            x != position(parent, splitdim) || throw3(x, splitdim)
        end
        local self, others
        let parent = parent
            self, others = box(parent, 0), ntuple(i->box(parent, i), Val{L})
        end
        parent.split = Split{p,T}(splitdims, xs, self, Children(others, metas))
        return others
    end
end
Box{p}(world::World{T,M}) where {p,T,M} = Box{p,T,M,calcL(Val(p))}(world)

"""
    root = Box{p}(world::World)

Create the root box of a CSp-tree for domain `world`.

    box = Box(parent::Box, splitdims::(Integer...), xs::(Real...), metas::(Any...))

Split a `parent` box along dimensions `splitdims`, where the length of
`splitdims` and `xs` is equal to `p`. The `2^p - 1` new evaluation
points have metadata `metas`, stored in "column-major" order
corresponding to positions

    (xs[1], xp[2], ...)  # (xp[1], xp[2], ...), the "self" box, is omitted
    (xp[1], xs[2], ...)
    (xs[1], xs[2], ...)

where `xp` is `position(parent)[splitdims]`.  For dimensions not
listed in `splitdims`, the positions of these children are all
identical to the parent evaluation point.
"""
Box(parent::Box{p,T,M,L}, splitdims::NTuple{p,Integer}, xs::NTuple{p,Real}, metas::NTuple{L,Any}) where {p,T,M,L} =
    Box{p,T,M,L}(parent, splitdims, xs, metas)

Box(parent::Box{1,T,M,1}, splitdim::Integer, x::Real, meta) where {T,M} =
    Box(parent, (splitdim,), (x,), (meta,))

isroot(box::Box) = box.parent == box
isleaf(box::Box) = !isdefined(box, :split)
isself(box::Box)  = !isroot(box) && box.parent.split.self == box
isother(box::Box) = !isroot(box) && box.parent.split.self != box
Base.eltype(::Type{B}) where B<:Box = B
Base.eltype(box::Box) = eltype(typeof(box))
Base.ndims(box::Box) = length(box.world.lower)
degree(::Type{B}) where B<:Box{p} where p = p
degree(box::Box) = degree(typeof(box))
boxcoordtype(::Type{B}) where B<:Box{p,T} where {p,T} = T
boxcoordtype(box::Box) = boxcoordtype(typeof(box))
maxchildren(::Type{Box{p,T,M,L}}) where {p,T,M,L} = L+1
maxchildren(::Type{B}) where B<:Box{p} where p = 1<<p  # for partial type like Box{2}
maxchildren(box::Box) = maxchildren(typeof(box))

function Base.show(io::IO, box::Box)
    print(io, "Box")
    showcompact(io, meta(box))
    print(io, "@", position(box))
end

# Compute 2^p-1 in a way that makes it inferrable
@generated function calcL(::Val{p}) where p
    return 1<<p - 1
end

## AbstractTrees interface

struct ChildrenIterator{B<:Box}
    box::B
end

AbstractTrees.children(box::Box) = ChildrenIterator(box)

Base.start(iter::ChildrenIterator) = 0
Base.done(iter::ChildrenIterator, s::Int) = isleaf(iter.box) | (s == maxchildren(iter.box))
function Base.next(iter::ChildrenIterator, s::Int)
    split = iter.box.split
    item = s == 0 ? split.self : split.others.children[s]
    return (item, s+=1)
end

AbstractTrees.printnode(io::IO, box::Box) = isleaf(box) ? print(io, box) : print(io, box.split.dims)

## Other iteration.

# The iterators in AbstractTrees are not sufficiently high-performance
# for our needs. Moreover we need customized visitation patterns.

Base.iteratorsize(::Type{<:Box}) = Base.SizeUnknown()

struct VisitorState{B<:Box}
    box::B
    childindex::Int
end

abstract type CSpTreeIterator end

Base.iteratorsize(::Type{<:CSpTreeIterator}) = Base.SizeUnknown()

struct LeafIterator{B<:Box} <: CSpTreeIterator
    root::B
end
Base.eltype(::Type{LeafIterator{B}}) where B<:Box = B

struct NonleafIterator{B<:Box} <: CSpTreeIterator
    root::B
end
Base.eltype(::Type{NonleafIterator{B}}) where B<:Box = B

struct SplitIterator{B<:Box} <: CSpTreeIterator
    base::B
end
Base.eltype(::Type{SplitIterator{B}}) where B<:Box{p,T,M,L} where {p,T,M,L} =
    Split{p,T,M,B,L}

maxchildren(splits::SplitIterator) = maxchildren(splits.base)

struct ClimbingState{B<:Box}
    box::B                  # current top-level node
    visited::Bool           # true if box.split has already been returned
    skipchildindex::UInt8   # index of branch from which we climbed
    childindex::Int         # index of branch we're currently exploring
    branchiter::NonleafIterator{B}  # iterator for the current branch
    branchstate::VisitorState{B}    # state for the current branch
end

# Create a state that marks `box`'s branch as having been visited
ClimbingState(box::Box, visited::Bool) = ClimbingState(box, visited, box.childindex)

function ClimbingState(box::B, visited::Bool, skipchildindex::Integer) where B<:Box
    iter = NonleafIterator(box)
    branchstate = VisitorState(box, maxchildren(box))
    if isroot(box)
        @assert(!visited)
        return ClimbingState(box, true, skipchildindex, maxchildren(box), iter, branchstate)
    end
    return ClimbingState(box.parent, false, skipchildindex, -1, iter, branchstate)
end

# Replace the branch state
ClimbingState(cstate::ClimbingState{B}, vstate::VisitorState) where B =
    ClimbingState(cstate.box, cstate.visited, cstate.skipchildindex,
                  cstate.childindex, cstate.branchiter, vstate)
