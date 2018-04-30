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
    if (isroot(box) && isleaf(box)) || n <= 2
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
    agesentinel = typemax(Int)            # this value signals "never seen"
    age = fill(agesentinel, neven, neven) # recency of edge, younger is "worse"
    for i = 1:neven
        age[i,i] = 0   # a dimension can't be paired with itself
    end
    nfilled = fill(1, neven)
    agecounter = 1
    for split in splits(box)
        d1, d2 = split.dims
        if age[d1, d2] == agesentinel
            age[d1, d2] = age[d2, d1] = agecounter
            nfilled[d1] += 1
            nfilled[d2] += 1
        end
        agecounter += 1
        all(nfilled .>= neven-1) && break
    end
    # Build the weighted graph and solve the matching problem
    pairgraph = BlossomV.Matching(neven)
    for j = 1:neven, i = 1:j-1
        if age[i,j] > 0
            cost = agecounter - min(age[i,j], agecounter) # older has lower cost
            BlossomV.add_edge(pairgraph, i-1, j-1, cost)
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
