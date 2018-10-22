## Minimum diagonal adjustment to a matrix to make it
## positive-definite, preserving the off-diagonal values.
#
# The problem is to factor Q into C*C', where each column of C
# has at most two non-zero entries. Consequently we have the
# equation
#
#     C_{ik}*C_{jk} = Q_{ij}
#
# where k is the column of C corresponding to (i,j).
# In addition to these paired columns, we also allow single-entry
# columns with the value c_i (note lowercase). Then we can write
#
#    (C*C')_{ii} = c_i^2 + \sum_k C_{ik}^2
#
# and we seek to minimize
#
#  P = \sum_i (C*C')_{ii}/|Q_{ii}|
#
# over the i with Q_{ii} != 0, subject to the constraint
#
#  (C*C')_{ii} >= |Q_{ii}|.
#
# Unlike the straight numerator, the scaled version used in P is
# invariant with respect to physical units.
#
# This minimization can be done analytically by introducing a ratio
#
#     r_k = C_{ik}/C_{jk}
#
# and computing the gradient of P wrt r.

"""
    Qp = possemidef(Q)

Return a variant of Q preserving the off-diagonal values and adjusting
the diagonals, as needed, to make the matrix positive semi-definite. Any
NaN entries of Q are treated as 0. Q is assumed to be symmetric and
only the lower triangle is used.
"""
function possemidef(Q::AbstractMatrix)
    ind1, ind2 = axes(Q)
    ind1 == ind2 || error("Matrix must be square")
    Qp = copy(Q)
    for i in ind1
        Qp[i,i] = 0
    end
    for j in ind2
        csum = zero(eltype(Q))
        qjj = abs(Q[j,j])
        for i = j+1:last(ind1)
            q = Q[i,j]
            ((q == 0) | isnan(q)) && continue
            qii = abs(Q[i,i])
            r = sign(q) * sqrt(qjj/qii)
            if r == 0 || !isfinite(r)
                r = oftype(r, sign(q))
            end
            Qp[j,j] += q * r
            Qp[i,i] += q / r
        end
    end
    for i in ind1
        # this is equivalent to adding c_i^2
        Qp[i,i] = max(Qp[i,i], abs(Q[i,i]))
    end
    return Qp
end
