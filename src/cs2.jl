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
        allfilled = true
        @inbounds @simd for i = 1:neven
            allfilled &= nfilled[i] >= neven-1
        end
        allfilled && break
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

"""
    c, g, Q, b = fit_quadratic(box)
    c, g, Q, b = fit_quadratic(f, box)

Return the coefficients of a quadratic model "centered" at `box`. The
model returns values

    m(x) = c + g'*Δx + (Δx'*Q*Δx)/2

where `Δx = x - b`.
"""
function fit_quadratic(f::Function, box::Box{2}; skip::Int=0)
    c, g, Q, b, y, prec, firstmatch = fit_quadratic_native(f, box; skip=skip)
    return canonicalize(c, g, Q, b, y, prec)
end
fit_quadratic(box::Box; skip::Int=0) = fit_quadratic(value, box; skip=skip)

"""
    c, g, Q, b, y, prec, firstmatch = fit_quadratic_native(box)
    c, g, Q, b, y, prec, firstmatch = fit_quadratic_native(f, box)

Return the coefficients of a CS2-native quadratic model "centered" at
`box`. The model returns values

    m(x) = c + g'*(x-b) + (1/2) ∑_{ij} (x[i] - b[i]) * Q[i,j] * (x[j] - χ[j,i])

where

             y_j          if i==j;
    χ[j,i] = b_j          if dimension i was split before or with j;
             2*y_j - b_j  if dimension i was split after j.

Precedence of splitting is determined with respect to `box`'s chain
(see [`CoordinateSplittingPTrees.chaintop`](@ref)).

See also [`CoordinateSplittingPTrees.fit_quadratic`](@ref).
"""
function fit_quadratic_native(f::Function, box::Box{2}; skip::Int=0)
    c, g, b, y, prec, firstmatch = fit_poly1(f, box; skip=skip)
    Q = coefficients_p(f, box; skip=skip)
    fit_poly2_diag!(f, Q, box, c, g, b, y, prec; skip=skip)
    return c, g, Q, b, y, prec, firstmatch
end
fit_quadratic_native(box::Box{2}; skip::Int=0) = fit_quadratic_native(value, box; skip=skip)

"""
    c, gc, Q, b = canonicalize(c, g, Q, b, y, prec)

Return the CS2-native quadratic model in canonical form,

    m(x) = c + gc'*Δx + (Δx'*Q*Δx)/2

where `Δx = x - b`.
"""
function canonicalize(c, g, Q, b, y, prec)
    gc = copy(g)
    n = length(g)
    for j = 1:n, i = 1:n
        χ = chi(j, i, b, y, prec)
        gc[i] += (b[j] - χ)*Q[i,j]/2
    end
    return c, gc, Q, b
end

function modelvalue(c, g, Q, b, y, prec::AbstractMatrix{Bool}, x)
    # For the CS2-native model
    n = length(x)
    val = c + g'*(x - b)
    for j = 1:n, i = j:n
        coef = CoordinateSplittingPTrees.Qcoef_value(i, j, x, b, y, prec)
        if coef != 0  # avoids using NaNs unless we need them
            val += Q[i,j]*coef*(1 + (i!=j))  # off-diagonals are done once
        end
    end
    return val
end

function modelvalue(c, g, Q, b, x)
    # For the canonical model
    Δx = x - b
    return c + g'*Δx + (Δx'*Q*Δx)/2
end

# Fit the first-order coefficients and set base positions
function fit_poly1(f::Function, box::Box{2,T}; skip::Int=0) where T
    box = getleaf(box)
    n = ndims(box)
    # Start at the top of the chain of splits as generated by `addpoint!`
    box, success = chaintop(box)
    success || error("no chain found at $box")

    b = position(box)
    y = similar(b)  # opposite point
    g = zeros(T, n)
    wassplit = falses(n)
    firstmatch = zeros(Int, n)
    # Rather than computing χ directly, store y and prec where
    # prec[i,j] is true if i ⪯ j. This avoids any risk of roundoff
    # error in the cancelation of Q coefficients (see Qcoef_value below).
    prec = fill(false, n, n)

    # Visit splits to cover each dimension once.
    # This suffices to determine the constant and gradient.
    c = f(box)
    npairs = ceil(Int, n/2)
    for paircounter = 1:npairs
        split = box.split
        d1, d2 = split.dims
        @assert(d1 != d2)
        @assert(!wassplit[d1] && (d2 > n || !wassplit[d2]))
        x1, x2 = split.xs
        f1, f2, f12 = f.(split.others.children)
        f0 = f(split.self)
        y[d1], wassplit[d1], firstmatch[d1] = x1, true, d2
        if d2 <= n
            y[d2], wassplit[d2], firstmatch[d2] = x2, true, d1
            prec[:, d2] = wassplit
        end
        prec[:, d1] = wassplit
        g[d1] = (f1 - f0)/(x1 - b[d1])
        if d2 <= n
            g[d2] = (f2 - f0)/(x2 - b[d2])
        end

        box = split.others.children[3]
    end
    return c, g, b, y, prec, firstmatch
end
fit_poly1(box) = fit_poly1(value, box)

function fit_poly2_diag!(f::Function, Q, box::Box{2}, c, g, b, y, prec; skip::Int=0)
    newinfo(x, b, y) = (x != b) & (x != y)
    function diag_equation(d, xnew, g, Q, x, b, y, prec)
        n = length(x)
        xd = x[d]
        coef = Qcoef_lineq(d, d, xnew-xd, xd, b[d], y[d], prec[d,d])
        consts = -g
        for k = 1:n
            k == d && continue  # already in coef
            xc = Qcoef_lineq(d, k, xnew-xd, x[k], b[k], y[k], prec[d,k])
            consts -= Q[d,k] * xc
        end
        return coef, consts
    end
    function process_split!(f, Q, checked, split, dimidx, x, g, b, y, prec)
        n = length(b)
        d = split.dims[dimidx]
        (d > n || checked[d]) && return 0
        bd = b[d]
        yd = y[d]
        xd = split.xs[dimidx]
        if newinfo(xd, bd, yd)
            coef, consts = diag_equation(d, xd, g[d], Q, x, b, y, prec)
            child = split.others.children[dimidx]
            ref = split.self
            Q[d,d] = (consts + (f(child)-f(ref))/(xd-x[d]))/coef
            checked[d] = true
            return 1
        end
        # Try it the other way too (we don't know which child we came from)
        if newinfo(x[d], bd, yd)
            xd, x[d] = x[d], xd
            d2 = split.dims[3-dimidx]
            xd2 = split.xs[3-dimidx]
            if d2 <= n
                xd2, x[d2] = x[d2], xd2
            end
            coef, consts = diag_equation(d, xd, g[d], Q, x, b, y, prec)
            child = split.others.children[3-dimidx]
            ref = split.others.children[3]
            Q[d,d] = (consts + (f(child)-f(ref))/(xd-x[d]))/coef
            checked[d] = true
            xd, x[d] = x[d], xd  # restore
            if d2 <= n
                xd2, x[d2] = x[d2], xd2
            end
            return 1
        end
        return 0
    end

    n = ndims(box)
    x = position(box)
    filled = similar(x, Bool)
    checked = falses(n)
    nremaining = n
    nsplits = 0
    for split in splits(box)
        nremaining <= 0 && break
        nsplits += 1
        nsplits <= skip && continue
        position!(x, filled, split.self)
        nremaining -= process_split!(f, Q, checked, split, 1, x, g, b, y, prec)
        nremaining -= process_split!(f, Q, checked, split, 2, x, g, b, y, prec)
    end
    return Q
end
fit_poly2_diag!(Q, box::Box{2}, c, g, b, y, prec) = fit_poly2_diag!(value, Q, box, c, g, b, y, prec)

function chi(j, i, b, y, prec)
    i == j && return y[j]
    return prec[i,j] ? b[j] : 2*y[j]-b[j]
end

function chi(b, y, prec)
    n = length(b)
    return [chi(j, i, b, y, prec) for j=1:n, i=1:n]
end

"""
    coef = Qcoef_value(i, j, x, b, y, prec)

Compute the symmetrized coefficient of `Q[i,j]` in the model value
expression.  The native expression is shown in the help for
[`CoordinateSplittingPTrees.fit_quadratic_native`](@ref). Note that
this expression is asymmetric. This function returns a symmetrized
version,

    ((x[i] - b[i])*(x[j] - χ[j,i]) + (x[j] - b[j])*(x[i] - χ[i,j]))/4

which is the mean of the coefficients for `Q[i,j]` and `Q[j,i]`. In
particular, it returns exactly 0 if the two should (analytically,
i.e. to infinite precision) cancel one another.
"""
function Qcoef_value(i, j, x, b, y, prec)
    i == j && return (x[i]-b[i])*(x[i]-y[i])/2
    bi, bj = b[i], b[j]
    yi, yj = y[i], y[j]
    xi, xj = x[i], x[j]
    pij, pji = prec[i, j], prec[j, i]
    (((pij & !pji) & (xi == yi)) | ((pji & !pij) & (xj == yj))) && return zero(xi*xj/2)
    χij = ifelse(pji, bi, 2yi-bi)
    χji = ifelse(pij, bj, 2yj-bj)
    return ((xi - bi)*(xj - χji) + (xj - bj)*(xi - χij))/4
end

function Qcoef_lineq(i, k, Δxi, xk::Real, bk::Real, yk::Real, precik::Bool)
    coef = zero(xk)
    if i == k
        (xk == bk && Δxi == yk-bk) || (xk == yk && Δxi == bk-yk) && return coef
        return Δxi/2 + xk - (bk+yk)/2
    end
    if precik
        xk == bk && return coef
        return coef + (xk - bk)
    end
    return coef + (xk - yk)
end


# An alternative to fit_poly1 and fit_poly2_diag! that accumulates 2x2
# equations. This does not depend on having chains, but it does mean
# that the gradient is dependent on all the off-diagonal elements of
# Q. (If some of those are missing, then g will be contaminated.)
function fit_quadratic_gdiag!(f::Function, Q, box::Box{2,T}, b=position(box); skip::Int=0) where T
    n = ndims(box)
    g = fill(T(NaN), n)
    qcoefs = fill(T(NaN), 2, n)
    rhs = fill(T(NaN), 2, n)
    filled = similar(g, Bool)
    x = position(box)
    nremaining = 2*n
    c = f(box)
    # Iterate until we fill all coefficients or exhaust the tree
    nsplits = 0
    for split in splits(box)
        nremaining == 0 && break
        nsplits += 1
        nsplits <= skip && continue
        for (i, d) in enumerate(split.dims)
            d <= n || continue
            if isnan(qcoefs[2,d])
                # There's at least one to fill
                position!(x, filled, split.self)
                xs = split.xs[i]
                xcoef = (x[d]+xs)/2 - b[d]
                xcoef == qcoefs[1,d] && continue  # not independent of the first
                idx = isnan(qcoefs[1,d]) ? 1 : 2
                Δx = xs - x[d]
                fd = (f(split.others.children[i]) - f(split.self))/Δx
                for j = 1:n
                    j == d && continue
                    fd -= Q[d,j]*(x[j] - b[j])
                end
                if isfinite(xcoef) && isfinite(fd)
                    qcoefs[idx,d] = xcoef
                    rhs[idx,d] = fd
                    nremaining -= 1
                end
            end
        end
    end
    # Solve the 2x2 equations
    for d = 1:n
        if !isnan(qcoefs[2,d])
            a12, a22 = qcoefs[1,d], qcoefs[2,d]  # a11 and a21 are both 1
            r1, r2 = rhs[1,d], rhs[2,d]
            det = a22 - a12    # manual inverse of 2x2 matrix
            g[d] = (a22*r1 - a12*r2)/det
            Q[d,d] = (-r1 + r2)/det
        end
    end
    return c, g, Q, b
end

fit_quadratic_gdiag!(Q, box::Box{2}, b=position(box); skip::Int=0) =
    fit_quadratic_gdiag!(value, Q, box, b; skip=skip)
