__precompile__(true)

module CoordinateSplittingPTrees

using AbstractTrees  # for display of tree structures
using Compat

export Box, World
export position, boxbounds, meta, value, degree, isroot, isleaf, getroot, getleaf, find_leaf_at
export leaves, splits, splitprint, splitprint_colored, print_tree

include("types.jl")
include("tree.jl")

"""
CoordinateSplittingPTrees implements coordinate-splitting trees of degree p
(CSp-trees). These trees are reminiscent of k-d trees, and are useful
for multidimensional full-degree polynomial interpolation.

The world bounds are specified by creating a [`World`](@ref) object.

The tree is initialized with `root = Box{p}(world)`, and new leaves are
added with `Box(parent, dims, xs, metas)` (see [`Box`](@ref)).

You can interact with boxes using the following API:

- [`position`](@ref): retrieve the position of a `Box`
- [`meta`](@ref): retrieve the metadata associated with a `Box`
- [`value`](@ref): retrieve the function value associated with a `Box`
- [`boxbounds`](@ref): retrieve the edges of a `Box`
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
for dimensions 2 and 4, etc. All `n` dimensions should appear in
`dimlists`; if `n` is not an integer multiple of `p`, then duplicates
should occur only in the final entry.

`metagen(y)` returns the metadata to be associated with position `y`.
"""
function addpoint!(box::Box, x, dimlists, metagen::Function)
    # Validate dimlists
    covered = falses(ndims(box))
    for i = 1:length(dimlists)-1
        dimlist = dimlists[i]
        for d in dimlist
            covered[d] && error("duplicate dimensions must be last, got $dimlists")
            covered[d] = true
        end
    end
    for d in dimlists[end]
        covered[d] = true
    end
    all(covered) || error("all dimensions must be covered, got $dimlists")

    if !isleaf(box)
        box = find_leaf_at(box, x)
    end
    b = position(box)
    y = copy(b)
    xc = copy(x)
    L = maxchildren(box)
    metas = typeof(meta(box))[]
    l = falses(degree(box))
    fill!(covered, false)
    for (k, dimlist) in enumerate(dimlists)
        dimlistv = [dimlist...]
        if k == length(dimlists)
            for d in dimlist
                if covered[d]
                    # For any duplicated dimensions we bisect the
                    # final interval to ensure it's different from
                    # either value.
                    xc[d] = (3*x[d] + b[d])/4
                end
            end
        else
            covered[dimlistv] = true
        end
        empty!(metas)
        for i = 1:L-1
            l.chunks[1] = i  # convert the counter into a logical bit pattern
            dl = dimlistv[l]
            y0 = y[dl]
            y[dl] = xc[dl]
            push!(metas, metagen(y))
            y[dl] = y0
        end
        box = Box(box, dimlist, (d->xc[d]).(dimlist), (metas...,))[end]
        l.chunks[1] = L-1
        dl = dimlistv[l]
        y[dl] = xc[dl]
    end
    return box
end

end # module
