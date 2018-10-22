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
splitprint(box::Box) = splitprint(stdout, box)

"""
    splitprint_colored([io::IO], box, innerbox)

Like [`splitprint`](@ref), except that `innerbox` is highlighted in red, and the chain
of parents of `innerbox` are highlighted in cyan.
"""
function splitprint_colored(io::IO, box::Box, thisbox::Box, allparents=get_allparents(thisbox))
    if isleaf(box)
        box == thisbox ? printstyled(io, 'l', color=:light_red) : print(io, 'l')
    else
        if box == thisbox
            printstyled(io, box.split.dims, color=:light_red)
        elseif box ∈ allparents
            printstyled(io, box.split.dims, color=:cyan)
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
splitprint_colored(box::Box, thisbox::Box) = splitprint_colored(stdout, box, thisbox)

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
Base.position(box::Box{p,T}) where {p,T} = position!(Vector{T}(undef, ndims(box)), box)

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
            i = findfirst(isequal(d), split.dims)
            if i !== nothing && ((childindex >> (i-1))&0x01) != 0x00
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
position!(x, box::Box) = position!(x, Vector{Bool}(undef, ndims(box)), box)

function position!(x, filled, box::Box)
    N = ndims(box)
    @assert(LinearIndices(x) == LinearIndices(filled) == Base.OneTo(N))
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
    bbs = Vector{Tuple{T,T}}(undef, ndims(box))
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
        i = findfirst(isequal(d), split.dims)
        if i !== nothing
            xs, xo = position(split.self, d), split.xs[i]
            x = position(box0, d)
            # @show p lower upper lfilled ufilled xs xo x
            xl, xu = xs < xo ? (xs, xo) : (xo, xs)
            # @show xl x xu
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
    lfilled = Vector{Bool}(undef, ndims(box))
    ufilled = Vector{Bool}(undef, ndims(box))
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
ncollinear(box::Box, top::Box=getroot(box)) = ncollinear!(Vector{Int}(undef, ndims(box)), Vector{Bool}(undef, ndims(box)), box, top)

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
    i = findfirst(isequal(splitdim), split.dims)
    _ncollinear!(nc, split.self, splitdim)
    i === nothing && return nc
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

getchild(split::Split, childidx) = childidx == 0 ? split.self :
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
    # Check that x is in the interior of root
    for i = 1:n
        bb = boxbounds(root, i)
        ((bb[1] <= x[i]) && (x[i] < bb[2] || (x[i] == bb[2] && bb[2] == root.world.upper[i]))) || throwbb(bb, x[i], i)
    end
    isleaf(root) && return root
    while !isleaf(root)
        split = root.split
        childidx = 0  # assign using bitwise arithmetic
        for j = 1:degree(root)
            sd, xother = split.dims[j], split.xs[j]
            sd > n && continue
            xself = position(root, sd)
            xsd = x[sd]
            # We need to choose the child where, for bb = boxbounds(child, sd),
            # we have bb[1] <= xsd < bb[2]. This is a requirement if we plan
            # to split child using the positions in x.
            #
            # Consequently this bit should be set according to
            #
            #                  |   xother > xself   |    xother < xself
            #    --------------|--------------------|---------------------
            #    xsd >= xmid   |         1          |           0
            #    --------------|--------------------|---------------------
            #    xsd < xmid    |         0          |           1
            #
            # where xmid = (xself+xother)/2.
            childidx |= ((xsd >= (xself+xother)/2) ⊻ (xother < xself)) << (j-1)
        end
        root = getchild(split, childidx)
    end
    return root
end


### Iteration

function Base.iterate(root::Box)
    isleaf(root) && return root, VisitorState(root, maxchildren(root))
    return root, VisitorState(root, 0)
end

function Base.iterate(root::Box, state::VisitorState)
    box, childindex = state.box, state.childindex
    if childindex < maxchildren(box) && !isleaf(box)
        # descend
        item = getchild(box.split, childindex)
        return item, VisitorState(item, isleaf(item) ? maxchildren(item) : 0)
    end
    # ascend
    while box != root
        childindex = box.childindex + 1
        box = box.parent
        if childindex < maxchildren(box)
            item = getchild(box.split, childindex)
            return item, VisitorState(item, isleaf(item) ? maxchildren(item) : 0)
        end
    end
    return nothing
end


"""
    iter = leaves(box)

Return an iterator visiting the leaf-nodes below `box` in depth-first order.
"""
leaves(root::Box) = Iterators.filter(box->isleaf(box) && !isfake(box), root)

"""
    isfake(box)

True if `box` has been displaced from its parent along a fictive dimension.
"""
function isfake(box::Box)
    n = ndims(box)
    p = box.parent
    isleaf(p) && return false
    split = p.split
    tf = false
    cidx = box.childindex
    for d in split.dims
        tf |= ((cidx & 0x01) == 0x01) & (d > n)
        cidx = cidx >> 0x01
    end
    return tf
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

function Base.iterate(splits::SplitIterator)
    box = splits.base
    if !isleaf(box)
        # Start at the current split if that's what was used to create the iterator
        branchhead = box.split.self
        branchiter = Iterators.filter(isnonleaf, branchhead)
        return box.split, ClimbingState(box, true, typemax(UInt8), 0, branchiter, VisitorState(branchhead, -1))
    end
    # The iterator was created from a leaf, which has no splits.
    # Go up and grab a sibling or parent.
    isroot(box) && return nothing
    skipchildindex, childindex = box.childindex, 0
    box = box.parent
    while nobranch(box, skipchildindex, childindex)
        childindex += 1
    end
    if childindex >= maxchildren(box)
        # This is a tree with only 1 split, so use the parent
        branchiter = Iterators.filter(isnonleaf, box)
        item, branchstate = iterate(branchiter)
        return item.split, ClimbingState(box, true, skipchildindex, childindex, branchiter, branchstate)
    end
    # Return a sibling
    branchhead = getchild(box.split, childindex)
    branchiter = Iterators.filter(isnonleaf, branchhead)
    item, branchstate = iterate(branchiter)  # this can't fail to return an item, because we already checked
    return item.split, ClimbingState(box, false, skipchildindex, childindex, branchiter, branchstate)
end

Base.iterate(splits::SplitIterator, ::Nothing) = iterate(splits)

function Base.iterate(splits::SplitIterator, state::ClimbingState)
    # First, try iterating on the existing branch and see if that returns a valid item
    branchiter, branchstate = state.branchiter, state.branchstate
    ret = branchstate.childindex < 0 ? iterate(branchiter) : iterate(branchiter, branchstate)
    if ret !== nothing
        box, branchstate = ret
        return box.split, ClimbingState(state, branchstate)
    end
    # The branch is done. Let's see if we can find another one.
    box = state.box
    skipchildindex, childindex = state.skipchildindex, state.childindex+1
    while nobranch(box, skipchildindex, childindex)
        childindex += 1
    end
    if childindex < maxchildren(box)
        # We still haven't exhausted the children of the current box
        branchhead = getchild(box.split, childindex)
        branchiter = Iterators.filter(isnonleaf, branchhead)
        item, branchstate = iterate(branchiter)  # this can't fail to return an item, because we already checked
        return item.split, ClimbingState(box, state.visited, skipchildindex, childindex, branchiter, branchstate)
    end
    if !state.visited
        # We haven't returned the current box's split, so do that now
        return box.split, ClimbingState(box, true, state.skipchildindex, childindex, branchiter, branchstate)
    end
    # Go up
    isroot(box) && return nothing
    skipchildindex = box.childindex
    box = box.parent
    childindex = 0
    while nobranch(box, skipchildindex, childindex)
        childindex += 1
    end
    if childindex >= maxchildren(box)
        return box.split, ClimbingState(box, true, state.skipchildindex, childindex, branchiter, branchstate)
    end
    branchhead = getchild(box.split, childindex)
    branchiter = Iterators.filter(isnonleaf, branchhead)
    item, branchstate = iterate(branchiter)  # this can't fail to return an item, because we already checked
    return item.split, ClimbingState(box, false, skipchildindex, childindex, branchiter, branchstate)
end

"""
    iter = chain(box)

Create an iterator that visits splits that minimally cover all dimensions
"""
chain(box::Box) = ChainIterator(box)

function Base.iterate(iter::ChainIterator)
    ndims(iter.base) == 0 && return nothing
    box, count = iter.base, 0
    item = box.split
    return item, (item.others.children[end], count+degree(box))
end

function Base.iterate(iter::ChainIterator, state)
    state[2] >= ndims(iter.base) && return nothing
    box, count = state
    item = box.split
    return item, (item.others.children[end], count+degree(box))
end

Base.length(iter::Union{Box,CSpTreeIterator}) = _length(iter)
Base.length(iter::Iterators.Filter{F,I}) where {F,I<:Union{Box,CSpTreeIterator}} =
    _length(iter)

function _length(iter)
    len = 0
    ret = iterate(iter)
    while ret !== nothing
        len += 1
        _, state = ret
        ret = iterate(iter, state)
    end
    len
end

function skipsplits(iter, skip)
    skip == 0 && return nothing
    _, state = iterate(iter)
    for i = 1:skip-1
        _, state = iterate(iter, state)
    end
    return state
end

# useful for debugging
function Base.show(io::IO, state::ClimbingState)
    println(io, "box $(state.box), visited $(state.visited), splitdims $(state.box.split.dims), skipidx $(state.skipchildindex), childidx $(state.childindex)")
    print(io, "branchiter $(state.branchiter), branch-is-done $(iterate(state.branchiter, state.branchstate)===nothing)")
end
