boxcoordtype(::Type{T}) where T<:AbstractFloat = T
boxcoordtype(::Type{T}) where T<:Real = Float64

# The overall domain of the problem, plus initial evaluation locations
# and the function value at the initial point. Every box within the
# world will hold a link to this structure
struct World{T<:Real,P,M}
    lower::Vector{T}      # lower bounds on parameters
    upper::Vector{T}      # upper bounds on parameters
    position::Vector{P}   # initial position data along each dimension
    meta::M               # metadata at baseposition (see below)

    function World{T,P,M}(lower, upper, position, meta) where {T,P,M}
        # Validate the inputs
        N = length(lower)
        length(lower) == length(upper) == length(position) == N ||
            throw(DimensionMismatch("lower, upper, and position must have length $N (got $(length(lower)), $(length(upper)), and $(length(position))"))
        for i = 1:N
            world_validate(position[i], lower[i], upper[i])
        end
        return new{T,P,M}(lower, upper, position, meta)
    end
end

"""
    world = World(lower::AbstractVector, upper::AbstractVector, position::AbstractVector, meta)

Return a structure representing the rectangular domain and
initialization of a CoordinateSplittingPTree.  `lower` and `upper` are
the bounds of the coordinates, `position` is a vector that contains
position information---in the simplest case this may be just the
coordinates of an initial point. `meta` is the metadata at `position`.

In general, `position[i]` can contain whatever information you require
about dimension `i`. Packages may specialize the behavior of `World`
for different types of information---see the source for details.
"""
function World(lower::AbstractVector, upper::AbstractVector, position::AbstractVector, meta)
    T = promote_type(eltype(lower), eltype(upper))
    for x in position
        T = promote_type(T, world_eltype(x))
    end
    Tb = boxcoordtype(T)
    positionb = world_ofeltype.(Tb, position)
    return World{Tb,eltype(positionb),typeof(meta)}(lower, upper, positionb, meta)
end

## API for extending World:
#  - `baseposition([T], position)` must return an actual position,
#    i.e., a vector of real-valued numbers that lie between `lower`
#    and `upper`. If `T` is supplied it must be used as the eltype of
#    the returned vector.
#  - `baseposition(position[i])` should return the position in dimension `i`.
#  - `world_eltype(position[i])` must return a suitable element type
#    to be used by `baseposition`
#  - `world_ofeltype(T, position[i])` must convert to world_eltype `T`
#  - `world_validate(position[i], lower[i], upper[i])` should throw an
#    error if `position[i]` is invalid, and otherwise return `nothing`.
#  - `world_newposition(x, position[i], lower[i], upper[i])` should
#    return a valid location different from `x`.

baseposition(x::Real) = x
baseposition(position::AbstractVector{<:Real}) = position
baseposition(::Type{T}, position::AbstractVector{<:Real}) where T<:Real =
    T[T(x) for x in position]
world_eltype(x::T) where T<:Real = T
world_ofeltype(::Type{T}, x::Real) where T<:Real = T(x)
function world_validate(x::Real, l::Real, u::Real)
    l <= x <= u || throw(ArgumentError("position $x is not within bounds [$l, $u]"))
    return nothing
end
function world_newposition(x, s::Real, l::Real, u::Real)
    if x != s
        return s
    end
    return x + 1 < u ? x + 1 :
           x - 1 >= l ? x - 1 : error("no valid value found. Consider using `partition_interval` to find a new point.")
end

# We're also going to "reserve" the behavior of World when position[i]
# is a 2-tuple (as used by QuadSplit.jl). Since tuples are Base types,
# doing it in this package avoids risk of type piracy.

baseposition(s::Tuple{Real,Real}) = s[1]
baseposition(::Type{T}, splits::AbstractVector{TT}) where {T,TT<:Tuple{Real,Real}} =
    T[s[1] for s in splits]
baseposition(splits::AbstractVector{TT}) where TT<:Tuple{Real,Real} =
    [s[1] for s in splits]

world_eltype(v::Tuple{Real,Real}) = promote_type(typeof(v[1]), typeof(v[2]))
world_ofeltype(::Type{T}, v::Tuple{Real,Real}) where T = (T(v[1]), T(v[2]))

function world_validate(s::Tuple{Real,Real}, l::Real, u::Real)
    s[1] != s[2] || throw(ArgumentError("split points must be distinct, got $s"))
    l <= s[1] <= u || throw(ArgumentError("position $(s[1]) is not within bounds [$l, $u]"))
    l <= s[2] <= u || throw(ArgumentError("position $(s[2]) is not within bounds [$l, $u]"))
    return nothing
end

world_newposition(x, s::Tuple{Real,Real}, l::Real, u::Real) = x == s[1] ? s[2] : s[1]


function World(lower::AbstractVector{T}, upper::AbstractVector{T}, splits::AbstractVector{Tuple{T,T}}, meta) where T<:Real
    Tb = boxcoordtype(T)
    return World{Tb, Tuple{Tb,Tb}, typeof(meta)}(lower, upper, splits, meta)
end

function World(lower::AbstractVector{Tl}, upper::AbstractVector{Tu}, splits::AbstractVector{Tuple{T,T}}, meta) where {Tl<:Real,Tu<:Real,T<:Real}
    Tb = boxcoordtype(promote_type(Tl,Tu,T))
    return World{Tb, Tuple{Tb,Tb}, typeof(meta)}(lower, upper, splits, meta)
end


Base.ndims(world::World) = length(world.position)
baseposition(world::World{T}) where T = baseposition(T, world.position)


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

mutable struct Box{p,T,M,L,P}
    world::World{T,P,M}      # the overall problem domain
    parent::Box{p,T,M,L,P}   # the node above this one
    childindex::UInt8        # which of parent's children is this? (0=self)
    split::Split{p,T,M,Box{p,T,M,L,P},L}  # undefined if this box is a leaf node

    function Box{p,T,M,L,P}(world::World{T,P,M}) where {p,T,M,L,P}
        p > 8 && error("degree must be less than or equal to 8 (change childindex::UInt8 for p>8)")
        root = new{p,T,M,L,P}(world)
        root.parent = root
        root.childindex = 0
        return root
    end

    function Box{p,T,M,L,P}(parent::Box{p,T,M,L,P}, splitdims::NTuple{p,Integer}, xs::NTuple{p,Real}, metas::NTuple{L,M}) where {p,T,M,L,P}
        function box(parent::Box{p,T,M,L,P}, childindex::Integer) where {p,T,M,L,P}
            return new{p,T,M,L,P}(parent.world, parent, childindex)
        end

        @noinline throw0(sd, mxd) = error("got split along dimension $sd, max allowed is $mxd")
        @noinline throw1(x, bb, sd) = error("out-of-bounds evaluation position $x along dimension $sd (bounds are $bb)")
        @noinline throw2(x, bb, u, sd) = error("cannot split at the upper edge $x along dimension $sd because the boxbounds $bb are not at the upper world edge $u")
        @noinline throw3(x, sd) = error("position $x is identical to the parent along $sd")

        @assert(isleaf(parent))
        # We allow fictive dimensions when needed to "round out"
        # splits. For example, if n is odd but p = 2, we allow a
        # dimension n+1 so that (n+1)/2 pairs can cover all dimensions.
        n = ndims(parent)
        maxdim = ceil(Int, n/p)*p
        for (splitdim,x) in zip(splitdims, xs)
            0 < splitdim <= maxdim || throw0(splitdim, maxdim)
            splitdim > n && continue
            bb = boxbounds(parent, splitdim)
            if (x < bb[1]) | (x > bb[2])
                throw1(x, bb, splitdim)
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
                throw2(x, bb, u, splitdim)
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
Box{p}(world::World{T,P,M}) where {p,T,M,P} = Box{p,T,M,calcL(Val(p)),P}(world)

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
Box(parent::Box{p,T,M,L,P}, splitdims::NTuple{p,Integer}, xs::NTuple{p,Real}, metas::NTuple{L,Any}) where {p,T,M,L,P} =
    Box{p,T,M,L,P}(parent, splitdims, xs, metas)

Box(parent::Box{1,T,M,1}, splitdim::Integer, x::Real, meta) where {T,M} =
    Box(parent, (splitdim,), (x,), (meta,))

# You can supply fewer than p dimensions, in which case we fill in
# with fictive dimensions
function Box(parent::Box{p,T,M,L,P}, splitdims::NTuple{k,Integer}, xs::NTuple{k,Real}, metas) where {p,T,M,k,L,P}
    0 < k <= p || throw(DimensionMismatch("got $k dimensions, max allowed is $p"))
    nmeta = 2^k-1
    length(metas) == nmeta || error("got $(length(metas)) metadatas for $k split dimensions, need $nmeta")
    n = ndims(parent)
    parentmeta = meta(parent)
    local splitdimspad, xspad, metaspad
    let n = n, nmeta = nmeta, parentmeta = parentmeta   # julia issue 15276
        splitdimspad = ntuple(Val(p)) do d
            d <= k ? Int(splitdims[d]) : n + d - k
        end
        xspad = ntuple(Val(p)) do d
            d <= k ? T(xs[d]) : zero(T)
        end
        metaspad = ntuple(Val(L)) do d
            i = d & nmeta
            return i == 0 ? parentmeta : metas[i]
        end
    end
    return Box{p,T,M,L,P}(parent, splitdimspad, xspad, metaspad)
end


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
maxchildren(::Type{Box{p,T,M,L,P}}) where {p,T,M,L,P} = L+1
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
