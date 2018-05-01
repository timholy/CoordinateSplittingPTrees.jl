### Display

"""
    splitprint([io::IO], box)

Print a representation of all boxes below `box`, using parentheses prefaced by a number `n`
to denote a split along dimension `n`, and `l` to represent a leaf.
"""
function splitprint(io::IO, box::Box)
    if isleaf(box)
        print(io, 'l')
    else
        print(io, box.split.dims, '[')
        splitprint(io, box.split.self)
        for c in box.split.others.children
            print(io, ", ")
            splitprint(io, c)
        end
        print(io, ']')
    end
end
splitprint(box::Box) = splitprint(STDOUT, box)

"""
    splitprint_colored([io::IO], box, innerbox)

Like [`splitprint`](@ref), except that `innerbox` is highlighted in red, and the chain
of parents of `innerbox` are highlighted in cyan.
"""
function splitprint_colored(io::IO, box::Box, thisbox::Box, allparents=get_allparents(thisbox))
    if isleaf(box)
        box == thisbox ? print_with_color(:light_red, io, 'l') : print(io, 'l')
    else
        if box == thisbox
            print_with_color(:light_red, io, box.split.dims)
        elseif box ∈ allparents
            print_with_color(:cyan, io, box.split.dims)
        else
            print(io, box.split.dims)
        end
        print(io, '[')
        splitprint_colored(io, box.split.self, thisbox, allparents)
        for c in box.split.others.children
            print(io, ", ")
            splitprint_colored(io, c, thisbox, allparents)
        end
        print(io, ']')
    end
end
splitprint_colored(box::Box, thisbox::Box) = splitprint_colored(STDOUT, box, thisbox)

function get_allparents(box)
    allparents = Set{typeof(box)}()
    p = box
    while !isroot(p)
        p = p.parent
        push!(allparents, p)
    end
    allparents
end


### Geometry

"""
    x = position(box)

Return the n-dimensional position vector `x` at which this box was evaluated.
"""
Base.position(box::Box{p,T}) where {p,T} = position!(Vector{T}(uninitialized, ndims(box)), box)

"""
    x = position(box, d::Integer)

Return the position of `box` in dimension `d`.
"""
function Base.position(box::Box, d::Integer)
    default = baseposition(box.world.position[d])
    while !isroot(box)
        p = box.parent
        childindex = box.childindex
        if !isself(box)
            split = p.split
            i = findfirst(split.dims, d)
            if i > 0 && ((childindex >> (i-1))&0x01) != 0x00
                return split.xs[i]
            end
        end
        box = p
    end
    return default
end

"""
    position!(x::AbstractVector, box)
    position!(x::AbstractVector, filled::AbstractVector{Bool}, box)

Fill `x` with the n-dimensional position of `box`. Including `filled` avoids allocation of temporaries.
"""
position!(x, box::Box) = position!(x, Vector{Bool}(uninitialized, ndims(box)), box)

function position!(x, filled, box::Box)
    N = ndims(box)
    @assert(linearindices(x) == linearindices(filled) == Base.OneTo(N))
    # Fill in default values (those corresponding to root)
    for i = 1:N
        x[i] = baseposition(box.world.position[i])
    end
    # Traverse the tree to see which entries have changed since root
    fill!(filled, false)
    nfilled = 0
    while !isroot(box) && nfilled < length(filled)
        p = box.parent
        split = p.split
        childindex = box.childindex
        for i = 1:degree(box)
            sd, xo = split.dims[i], split.xs[i]
            sd > ndims(box) && continue  # fictive dimension
            if ((childindex >> (i-1))&0x01) != 0x00 && !filled[sd]
                x[sd] = xo
                filled[sd] = true
                nfilled += 1
            end
        end
        box = p
    end
    return x
end

"""
    bbs = boxbounds(box)

Compute the bounds (edge positions) of `box`. `bbs[d] = (lower, upper)` for dimension `d`.
"""
function boxbounds(box::Box{p,T}) where {p,T}
    bbs = Vector{Tuple{T,T}}(uninitialized, ndims(box))
    return boxbounds!(bbs, box)
end

"""
    lower, upper = boxbounds(box, d::Integer)

Compute the bounds of `box` along dimension `d`.
"""
function boxbounds(box::Box, d::Integer)
    lower, upper = box.world.lower[d], box.world.upper[d]
    box0 = box
    lfilled, ufilled = false, false
    while !isroot(box) && !(lfilled & ufilled)
        p = box.parent
        split = p.split
        i = findfirst(split.dims, d)
        if i > 0
            xs, xo = position(split.self, d), split.xs[i]
            x = position(box0, d)
            # @show p lower upper lfilled ufilled xs xo x
            xl, xu = xs < xo ? (xs, xo) : (xo, xs)
            if xl <= x <= xu
                if !lfilled && xl < x
                    lower, lfilled = max((xl+xu)/2, (xl+x)/2), true
                elseif !ufilled && xu > x
                    upper, ufilled = min((xl+xu)/2, (x+xu)/2), true
                end
            end
        end
        box = p
    end
    lower, upper
end

"""
    boxbounds!(bbs::AbstractVector, box)

Fill `bbs` with the bounds (edge positions) of `box`.
"""

function boxbounds!(bbs, box::Box)
    lfilled = Vector{Bool}(uninitialized, ndims(box))
    ufilled = Vector{Bool}(uninitialized, ndims(box))
    boxbounds!(bbs, lfilled, ufilled, box)
end

function boxbounds!(bbs, lfilled, ufilled, box::Box)
    fill!(lfilled, false)
    fill!(ufilled, false)
    n = ndims(box)
    for d = 1:n
        bbs[d] = (box.world.lower[d], box.world.upper[d])
    end
    box0 = box
    nlfilled = nufilled = 0
    while !isroot(box) && min(nlfilled, nufilled) < n
        p = box.parent
        split = p.split
        for (sd, xo) = zip(split.dims, split.xs)
            sd > n && continue   # fictive dimensions
            if !lfilled[sd] || !ufilled[sd]
                xs = position(split.self, sd)
                x = position(box0, sd)
                xl, xu = xs < xo ? (xs, xo) : (xo, xs)
                if xl <= x <= xu
                    if !lfilled[sd] && xl < x
                        bbs[sd], lfilled[sd] = (max((xl+xu)/2, (xl+x)/2), bbs[sd][2]), true
                        nlfilled += 1
                    elseif !ufilled[sd] && xu > x
                        bbs[sd], ufilled[sd] = (bbs[sd][1], min((xl+xu)/2, (x+xu)/2)), true
                        nufilled += 1
                    end
                end
            end
        end
        box = p
    end
    bbs
end

"""
    nc = ncollinear(box, top=getroot(box))

`nc[i]` is the number of points contained inside `top` that are
collinear with `position(box)` along dimension `i`. Being collinear
along dimension `i` implies that `position(otherbox, j) == position(box, j)`
for any dimension `j != i`.
"""
ncollinear(box::Box, top::Box=getroot(box)) = ncollinear!(Vector{Int}(uninitialized, ndims(box)), Vector{Bool}(uninitialized, ndims(box)), box, top)

function ncollinear!(nc::AbstractVector{Int}, filled::AbstractVector{Bool}, box::Box, top::Box=getroot(box))
    fill!(nc, 0)
    fill!(filled, false)
    nfilled = 0
    while nfilled < 2 && box != top
        split = box.parent.split
        childindex = cindex = box.childindex
        wasbase = nfilled == 0
        for i = 1:degree(box)
            sd = split.dims[i]
            otherindex = childindex ⊻ (0x01 << (i-1))
            if cindex&0x01 == 0x00 # if this is the same as the parent point
                (wasbase || filled[sd]) && _ncollinear!(nc, split.others.children[otherindex], sd)
            else  # if we go up with shift relative to the basepoint
                otherchild = otherindex == 0x00 ? split.self : split.others.children[otherindex]
                (wasbase || filled[sd]) && _ncollinear!(nc, otherchild, sd)
                nfilled += !filled[sd]
                filled[sd] = true
            end
            cindex = cindex >> 1
        end
        box = box.parent
    end
    return nc
end

function _ncollinear!(nc, box, splitdim)
    if isleaf(box)
        nc[splitdim] += 1
        return nc
    end
    split = box.split
    i = findfirst(split.dims, splitdim)
    _ncollinear!(nc, split.self, splitdim)
    i == 0 && return nc
    childidx = 1 << (i-1)
    _ncollinear!(nc, split.others.children[childidx], splitdim)
end


function epswidth(bb::Tuple{T,T}) where T<:AbstractFloat
    w1 = isfinite(bb[1]) ? eps(bb[1]) : T(0)
    w2 = isfinite(bb[2]) ? eps(bb[2]) : T(0)
    return max(w1, w2)
end
epswidth(bb::Tuple{Real,Real}) = epswidth(Float64.(bb))

### Traversal

"""
    root = getroot(box)

Return the root node for `box`.
"""
function getroot(box::Box)
    while !isroot(box)
        box = box.parent
    end
    return box
end

"""
    leaf = getleaf(box)

Get the leaf node at the position of `box`.
"""
function getleaf(box::Box)
    while !isleaf(box)
        box = box.split.self
    end
    return box
end

"""
    m = meta(box)

Return the metadata associated with `box`.
"""
function meta(box::Box)
    while isself(box)
        box = box.parent
    end
    isroot(box) && return box.world.meta
    @assert(box.childindex > 0)
    return box.parent.split.others.metas[box.childindex]
end

getchild(split, childidx) = childidx == 0 ? split.self :
                                            split.others.children[childidx]

# """
#     boxp = find_parent_with_splitdim(box, splitdim::Integer)

# Return the first node at or above `box` who's parent box was split
# along dimension `splitdim`. If `box` has not yet been split along
# `splitdim`, returns the root box.
# """
# function find_parent_with_splitdim(box::Box, splitdim::Integer)
#     while !isroot(box)
#         p = box.parent
#         if p.split.dim == splitdim
#             return box
#         end
#         box = p
#     end
#     return box
# end

"""
    box = find_leaf_at(root, x)

Return the leaf-node `box` beneath `root` that contains `x`.
"""
function find_leaf_at(root::Box, x)
    @noinline throwbb(bb, x, dim) = error("$x not within $bb along dimension $dim")
    @noinline throwdmm(n, l) = throw(DimensionMismatch("tree has $n dimensions, got $l"))
    n = ndims(root)
    length(x) == n || throwdmm(n, length(x))
    for i = 1:n
        bb = boxbounds(root, i)
        ((bb[1] <= x[i]) && (x[i] < bb[2] || (x[i] == bb[2] && bb[2] == root.world.upper[i]))) || throwbb(bb, x[i], i)
    end
    isleaf(root) && return root
    while !isleaf(root)
        split = root.split
        childidx = 0
        for j = 1:degree(root)
            sd, xother = split.dims[j], split.xs[j]
            sd > n && continue
            xself = position(root, sd)
            xsd = x[sd]
            childidx |= (abs(xsd - xother) <= abs(xsd - xself)) << (j-1)
        end
        root = getchild(split, childidx)
    end
    return root
end


### Iteration

Base.start(root::Box) = VisitorState(root, 0)
Base.done(root::Box, state::VisitorState) = state.childindex >= maxchildren(root)

function Base.next(root::Box, state::VisitorState)
    item, childindex = state.box, state.childindex
    # Depth-first search, visiting the parent (container) node before
    # visiting self & other children
    if isleaf(item)
        # Since we're at a leaf, for the next item we must go up the tree
        box, childindex = up(item, root)
        if 0 < childindex < maxchildren(root)
            return (item, VisitorState(box.split.others.children[childindex], 0))
        end
        @assert(box == root && childindex > 0)
        return (item, VisitorState(root, maxchildren(root)))
    end
    return (item, VisitorState(item.split.self, 0))
end

function up(box, root)
    # println("starting up with $box")
    box == root && return (box, maxchildren(root))
    local childindex
    while true
        box, childindex = box.parent, box.childindex
        # println("  to parent $box with childindex $childindex")
        # If this was the last of the parent's children, keep going up
        (box == root || childindex < maxchildren(root)-1) && break
    end
    # println("  isroot(box) = $(isroot(box))")
    return (box, childindex+1)
end


"""
    iter = leaves(box)

Return an iterator visiting the leaf-nodes below `box` in depth-first order.
"""
leaves(root::Box) = LeafIterator(root)

## LeafIterator

function Base.start(iter::LeafIterator)
    isleaf(iter.root) && return VisitorState(iter.root, 0)
    find_next_leaf(iter, VisitorState(iter.root, 0))
end
Base.done(iter::LeafIterator, state::VisitorState) = state.childindex >= maxchildren(iter.root)

function Base.next(iter::LeafIterator, state::VisitorState)
    @assert(isleaf(state.box))
    return (state.box, find_next_leaf(iter, state))
end
function find_next_leaf(iter::LeafIterator, state::VisitorState)
    _, state = next(iter.root, state)
    while !isleaf(state.box) && state.childindex < maxchildren(iter.root)
        _, state = next(iter.root, state)
    end
    return state
end

## NonleafIterator

# depth-first pre-order
function Base.start(iter::NonleafIterator)
    return VisitorState(iter.root, isleaf(iter.root) ? maxchildren(iter.root) : 0)
end

Base.done(iter::NonleafIterator, state::VisitorState) = state.childindex >= maxchildren(iter.root)

function Base.next(iter::NonleafIterator, state::VisitorState)
    @assert(!isleaf(state.box))
    return (state.box, find_next_nonleaf(iter, state))
end
function find_next_nonleaf(iter::NonleafIterator, state::VisitorState)
    _, state = next(iter.root, state)
    while isleaf(state.box) && state.childindex < maxchildren(iter.root)
        _, state = next(iter.root, state)
    end
    return state
end


"""
    iter = splits(box)

Return an iterator for visiting the dimension-splits in the sub-tree
containing `box`. Branches below `box` are explored first; once
exhausted, a step is taken up the tree and branches below the other
children of `box`'s parent are explored. Search terminates when all
branches below the root node have been explored.

The iterator returns the `split` field of non-leaf boxes (see
[`CoordinateSplittingPTrees.Split`](@ref)), so a loop should be written as

    for split in splits(box)
        # do something with split
    end
"""
splits(box::Box) = SplitIterator(box)

function nobranch(box, skipchildindex, childindex)
    return childindex < maxchildren(box) &&
        (childindex == skipchildindex ||
         isleaf(getchild(box.split, childindex)))
end

function Base.start(splits::SplitIterator)
    box = splits.base
    isleaf(box) && return ClimbingState(box, false)
    return ClimbingState(box.split.self, false, typemax(box.childindex))  # don't skip any of the children
end

function Base.next(splits::SplitIterator, state::ClimbingState)
    split, visited = state.box.split, state.visited
    item, state = _next(splits, state)
    if item == split && visited
        item, state = _next(splits, state)
    end
    return item, state
end

function _next(splits, state)
    branchiter, branchstate = state.branchiter, state.branchstate
    if !done(branchiter, branchstate)
        box, branchstate = next(branchiter, branchstate)
        return box.split, ClimbingState(state, branchstate)
    end
    box = state.box
    item = box.split
    # Find the next valid split
    skipchildindex, childindex = state.skipchildindex, state.childindex+1
    while nobranch(box, skipchildindex, childindex)
        childindex += 1
    end
    if childindex >= maxchildren(box)  # go up
        return item, ClimbingState(box, state.visited)
    end
    iter = NonleafIterator(getchild(box.split, childindex))
    return item, ClimbingState(box, true, skipchildindex, childindex, iter, start(iter))
end

function Base.done(splits::SplitIterator, state::ClimbingState)
    (isroot(state.box) & state.visited) || return false
    done(state.branchiter, state.branchstate) || return false
    state.childindex >= maxchildren(splits)-1 && return true
    split = state.box.split
    for i = state.childindex+1:maxchildren(splits)-1
        i == state.skipchildindex && continue
        isleaf(getchild(split, i)) || return false
    end
    return true
end


function Base.length(iter::Union{Box,CSpTreeIterator})
    state = start(iter)
    len = 0
    while !done(iter, state)
        _, state = next(iter, state)
        len += 1
    end
    len
end

# useful for debugging
function Base.show(io::IO, state::ClimbingState)
    println(io, "box $(state.box), visited $(state.visited), splitdims $(state.box.split.dims), skipidx $(state.skipchildindex), childidx $(state.childindex)")
    print(io, "branchiter $(state.branchiter.root), branch-is-done $(done(state.branchiter, state.branchstate))")
end
