# CoordinateSplittingPTrees

[![Build Status](https://travis-ci.org/timholy/CoordinateSplittingPTrees.jl.svg?branch=master)](https://travis-ci.org/timholy/CoordinateSplittingPTrees.jl)

[![codecov.io](http://codecov.io/github/timholy/CoordinateSplittingPTrees.jl/coverage.svg?branch=master)](http://codecov.io/github/timholy/CoordinateSplittingPTrees.jl?branch=master)

This repository implements a new (?) data structure, the *coordinate
splitting tree of degree p* or CSp-trees for short.  As shown
[here](notyetwritten), this data structure supports efficient and accurate
full-degree multidimensional polynomial interpolation. For example, in
`n` dimensions, a quadratic polynomial has `(n+1)*(n+2)/2`
coefficients; using traditional methods, solving the linear equations
that determine the coefficients from the data is an `O(n^6)`
operation, but a CS1-tree reduces that to `~O(n^4)` and a CS2-tree
reduces it to `O(n^2)`.  Moreover, the resulting coefficients are
determined with significantly higher precision than by traditional
approaches.  CSp trees naturally implement adaptive mesh refinement
(AMR), as adding new evaluation points to the interpolation scheme
corresponds to adding new nodes to the tree. This package is written
in the [Julia programming language](https://julialang.org/).

Full-degree multidimensional polynomial interpolation differs from
most other interpolation schemes in that it is not, in general,
continuous over space. Instead, the goal is to find a local polynomial
fit of full degree and use this for detailed investigations of local
structure---in essence, prioritizing accuracy of local interpolation
over any global property like continuity or smoothness. An important
application is derivative-free optimization (DFO), where many
algorithms work by constructing a quadratic fit to the local function
values and then minimizing the quadratic over some trust
region---since the goal is to find the location of the minimum (or
minima), global continuity of the interpolation scheme is therefore
not an urgent priority.  Specifically, if one has collected function
values at more than `(n+1)*(n+2)/2` points, then one requires a rule
to select a "complete" (with sufficient independence to determine the
coefficients) subset of evaluation positions to determine the model
coefficients. If different point subsets are selected in different
regions of space (e.g., choosing the nearest `(n+1)*(n+2)/2` points),
the polynomial will change discontinuously as one moves around the space.

CSp-trees have a close relationship with
[k-d trees](https://en.wikipedia.org/wiki/K-d_tree), and indeed CS1
trees can be implemented essentially as a k-d tree with only one
evaluation point (and substantial restrictions on its position) per
box. Specifically, a k-d tree "splits" space along one dimension at a
time; a CS1 tree selects a new evaluation point that differs from its
"parent" position in only one coordinate, thus creating a new box that
splits the parent along this coordinate. A general CSp tree splits `p`
dimensions simultaneously, creating a set of `2^p-1` new boxes each
associated with a unique position and new function evaluation.

For now, to learn how to use this pacakge just type
`?CoordinateSplittingPTrees` at the Julia prompt, and then read the
help associated with each function. For convenience, that overall
summary help-text is reproduced below.

### Usage summary (more detail available in the help for each function)

The world bounds are specified by creating a `World` object (see `?World`).

The tree is initialized with `root = Box{p}(world)`, and new leaves are
added with `Box(parent, dims, xs, metas)` (see `?Box`).

You can interact with boxes using the following API:

#### Polynomial fits

- `polynomial_full`: construct a full polynomial of the specified degree
- `polynomial_minimal`: construct a minimal polynomial of the specified degree

#### Tree API

- `position`: retrieve the position of a `Box`
- `meta`: retrieve the metadata associated with a `Box`
- `boxbounds`: retrieve the edges of a `Box`
- `isroot`: determine whether a given `Box` is the root-node
- `isleaf`: determine whether a given `Box` is a leaf-node
- `getroot`: get the root node associated with a tree
- `getleaf`: get the leaf node associated with a (possibly-parent) box
- `find_leaf_at`: find the leaf node containing a particular position

- `leaves`: returns an iterator for visiting all leaf-nodes

#### Display

- `print_tree`: display the tree structure (from AbstractTrees.jl)
- `splitprint`: a more compact display of the tree structure
- `splitprint_colored`: similar to `splitprint` but with highlights
