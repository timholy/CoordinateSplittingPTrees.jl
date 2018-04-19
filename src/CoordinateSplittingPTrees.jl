__precompile__(true)

module CoordinateSplittingPTrees

using AbstractTrees
using Compat

export Box, World
export position, meta, boxbounds, degree, isroot, isleaf, getroot, getleaf, find_leaf_at
export leaves, splitprint, splitprint_colored, print_tree

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

end # module
