__precompile__(true)

module CoordinateSplittingPTrees

using AbstractTrees  # for display of tree structures
using Compat

export Box, World
export position, boxbounds, meta, value, addpoint!
export getroot, getleaf, find_leaf_at, degree, isroot, isleaf
export leaves, splits, splitprint, splitprint_colored, print_tree

include("types.jl")
include("tree.jl")
include("polynomials.jl")

"""
CoordinateSplittingPTrees implements coordinate-splitting trees of
degree `p` (CSp-trees). These trees are reminiscent of [k-d
trees](https://en.wikipedia.org/wiki/K-d_tree), and are useful for
multidimensional full-degree polynomial interpolation.

The world bounds are specified by creating a [`World`](@ref) object.

The tree is initialized with `root = Box{p}(world)`, and new leaves are
added with `Box(parent, dims, xs, metas)` (see [`Box`](@ref)).

You can interact with boxes using the following API:

- [`position`](@ref): retrieve the position of a `Box`
- [`meta`](@ref): retrieve the metadata associated with a `Box`
- [`value`](@ref): retrieve the function value associated with a `Box`
- [`boxbounds`](@ref): retrieve the edges of a `Box`
- [`addpoint!`](@ref): split boxes to generate new positions
- [`isroot`](@ref): determine whether a given `Box` is the root-node
- [`isleaf`](@ref): determine whether a given `Box` is a leaf-node
- [`getroot`](@ref): get the root node associated with a tree
- [`getleaf`](@ref): get the leaf node associated with a (possibly-parent) box
- [`find_leaf_at`](@ref): find the leaf node containing a particular position

- [`leaves`](@ref): returns an iterator for visiting all leaf-nodes

- [`print_tree`](@ref): display the tree structure (from AbstractTrees.jl)
- [`splitprint`](@ref): a more compact display of the tree structure
- [`splitprint_colored`](@ref): similar to `splitprint` but with highlights
"""
CoordinateSplittingPTrees

"""
    z = value(box)

Retreive the function value associated with `box`. `value` calls
[`meta`](@ref), so whatever type `M` you use for metadata needs to
have a `value(::M)` function defined. `value(z::Real) = z` is
pre-defined.
"""
value(box::Box) = value(meta(box))
value(z::Real) = z

Base.isless(b1::Box, b2::Box) = isless(value(b1), value(b2))

Base.minimum(box::Box) = minimum(leaves(box))
Base.maximum(box::Box) = maximum(leaves(box))
Base.extrema(box::Box) = extrema(leaves(box))

"""
    addpoint!(box::Box, x, dimlists, metagen::Function)

Create new evaluation points by splitting existing boxes. `x`, the
final evaluation position, must be inside `box`. Dimensions are split
in the order specified by `dimlists`, which should be a
list-of-dimension-lists. For example, for a CS2 tree one might use

    dimlists = [(1,3), (2,4), (5,9), ...]

to indicate that the first split is for dimensions 1 and 3, the second
for dimensions 2 and 4, etc. All `n` dimensions should appear once in
`dimlists`; if `n` is not an integer multiple of `p`, then the final
entry should contain only the "leftover" dimensions. (Fictive
dimension(s) will be added automatically to make all splits of length
`p`.  If fictive dimensions are being used, some leaf-nodes will
appear to be duplicates of others, but in reality they are "separated"
(with size 0) along fictive dimension(s), which are not displayed, and
share the same metadata as other boxes with the same position in the
non-fictive dimensions.)

`metagen(y)` must return the metadata to be associated with position `y`,
where `y` is a vector.
"""
function addpoint!(box::Box, x, dimlists, metagen::Function)
    # Validate dimlists
    covered = falses(ndims(box))
    for dimlist in dimlists
        for d in dimlist
            covered[d] && error("dimension $d was duplicated")
            covered[d] = true
        end
    end
    all(covered) || error("all dimensions must be covered, got $dimlists")

    if !isleaf(box)
        box = find_leaf_at(box, x)
    end
    b = position(box)
    y = copy(b)
    L = maxchildren(box)-1
    metas = Vector{typeof(meta(box))}(uninitialized, L)
    l = falses(degree(box))
    for dimlist in dimlists
        dimlistv = [dimlist...]
        empty!(metas)
        length(dimlist) < degree(box) && resize!(l, length(dimlist))
        for i = 1:2^length(dimlist)-1
            l.chunks[1] = i  # convert the counter into a logical bit pattern
            dl = dimlistv[l]
            y0 = y[dl]
            y[dl] = x[dl]
            push!(metas, metagen(y))
            y[dl] = y0
        end
        box = Box(box, dimlist, (d->x[d]).(dimlist), (metas...,))[end]
        y[dimlistv] = x[dimlistv]
    end
    return box
end

end # module
