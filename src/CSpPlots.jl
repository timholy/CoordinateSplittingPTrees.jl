module CSpPlots

using CoordinateSplittingPTrees
using Makie, GeometryTypes, Colors, PerceptualColourMaps

export plotboxes

get_finite(x, default) = isfinite(x) ? x : default
outerbounds(b1, b2) = (min(b1[1], b2[1]), max(b1[2], b2[2]))
outerbounds(b1, b2::Real) = (min(b1[1], b2), max(b1[2], b2))
outerbounds_finite(b1, b2) = (min(b1[1], get_finite(b2[1], b1[1])), max(b1[2], get_finite(b2[2], b1[2])))
widenbounds(b) = (Δx = 0.1*(b[2]-b[1]); return (b[1]-Δx, b[2]+Δx))

function finitebounds(root::Box{p,T}) where {p,T}
    @assert(ndims(root) == 2)
    xbounds, ybounds = boxbounds(root)
    if all(isfinite, xbounds) && all(isfinite, ybounds)
        return widenbounds(xbounds), widenbounds(ybounds)
    end
    # Get bounds needed to contain all finite positions inside `root`
    xbounds, ybounds = (Inf, -Inf), (Inf, -Inf)
    lower, upper = root.world.lower, root.world.upper
    bb = Vector{Tuple{T,T}}(undef, 2)
    x = [NaN, NaN]
    lfilled, rfilled = [false, false], [false, false]
    for box in leaves(root)
        for i = 1:2
            bb[i] = (lower[i], upper[i])
        end
        CoordinateSplittingPTrees.boxbounds!(bb, lfilled, rfilled, box)
        xbounds = outerbounds_finite(xbounds, bb[1])
        ybounds = outerbounds_finite(ybounds, bb[2])
        x = CoordinateSplittingPTrees.position!(x, lfilled, box)
        xbounds = outerbounds(xbounds, x[1])
        ybounds = outerbounds(ybounds, x[2])
    end
    widenbounds(xbounds), widenbounds(ybounds)
end

function plotboxes(root::Box, xbounds::Tuple{Any,Any}, ybounds::Tuple{Any,Any};
                   clim=value.(extrema(root)),
                   cs::AbstractVector{<:Colorant}=cmap("RAINBOW3"))
    @assert(ndims(root) == 2)
    scene = Scene()
    rects, cols, points = Rectangle{Float32}[], eltype(cs)[], Point{2,Float32}[]
    for bx in leaves(root)
        addrect!(rects, cols, points, bx, clim, cs, xbounds, ybounds)
    end
    poly(rects, color=cols, linecolor=:black)
    scatter(points, color=:black, markersize=0.03)
    center!(scene)
    nothing
end
plotboxes(root::Box;
          clim=value.(extrema(root)),
          cs::AbstractVector{<:Colorant}=cmap("RAINBOW3")) =
    plotboxes(root, finitebounds(root)...; clim=clim, cs=cs)

function addrect!(rects, cols, points, box::Box, clim, cs, xbounds, ybounds)
    @assert(isleaf(box))
    @assert(ndims(box) == 2)
    x = position(box)
    bbx, bby = boxbounds(box)
    bbx, bby = clamp.(bbx, xbounds...), clamp.(bby, ybounds...)
    rect = Rectangle{Float32}(bbx[1], bby[1], bbx[2]-bbx[1], bby[2]-bby[1])
    fval = value(box)
    if isfinite(fval)
        push!(rects, rect)
        cnorm = (clamp(fval, clim...) - clim[1])/(clim[2] - clim[1])
        col = cs[round(Int, (length(cs)-1)*cnorm) + 1]
        push!(cols, col)
    end
    push!(points, Point{2,Float32}(x[1], x[2]))
    return rects, cols, points
end

end
