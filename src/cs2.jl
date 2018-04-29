"""
    dimlists = choose_dimensions(box::Box)

Choose combinations of dimensions that have been least recently used
for splitting in the vicinity of `box`. Passing these to
[`addpoint!`](@ref) will enable fresh estimates of the
corresponding coefficients of the polynomial.
"""
function choose_dimensions(box::Box{2})
    pairs = Union{Tuple{Int,Int},Tuple{Int}}[]
    n = ndims(box)
    if isroot(box) && isleaf(box)
        for i = 1:2:n
            if i < n
                push!(pairs, (i, i+1))
            else
                push!(pairs, (i,))
            end
        end
        return pairs
    end
    # Determine the least recently seen splits in the vicinity of
    # `box`.  We iterate over the splits until at least one dimension
    # has only one unseen pairing left. Allowing only unseen pairings,
    # we then construct a maximal matching
    # (https://en.wikipedia.org/wiki/Matching_(graph_theory))
    neven = n + (isodd(n) ? 1 : 0)
    Qflag = falses(neven, neven)
    for i = 1:neven
        Qflag[i,i] = true   # a dimension can't be paired with itself
    end
    nfilled = fill(1, neven)
    for split in splits(box)
        d1, d2 = split.dims
        if !Qflag[d1, d2]
            Qflag[d1, d2] = Qflag[d2, d1] = true
            nfilled[d1] += 1
            nfilled[d2] += 1
        end
        any(nfilled .>= neven-1) && break
    end
    # Build a graph where the edges are the missing entries in Qflag
    # and solve the matching problem
    pairgraph = BlossomV.Matching(n + isodd(n)) # BlossomV requires even # nodes
    for j = 1:neven, i = 1:j-1
        if !Qflag[i,j]
            BlossomV.add_edge(pairgraph, i-1, j-1, 1)
        end
    end
    if neven > n && !any(@views Qflag[:,end])
        # if necessary add an edge from every real node to fictitious node
        for i = 1:n
            BlossomV.add_edge(pairgraph, i-1, n, 2)
        end
    end
    BlossomV.solve(pairgraph)
    seen = falses(n)
    extra = 0  # if a real node is paired with a fictitious node, add it last
    for i = 1:n
        seen[i] && continue
        j = BlossomV.get_match(pairgraph, i-1) + 1
        if j <= n
            if !seen[j]
                push!(pairs, (i, j))
            end
            seen[i] = seen[j] = true
        else
            extra = i
        end
    end
    if extra != 0
        push!(pairs, (extra,))
    end
    return pairs
end
