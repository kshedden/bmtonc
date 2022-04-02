cols = ["blue", "lightblue", "grey", "orange", "red"]

# Rotate so that the scores for different factors are orthogonal.
# 'u' contains the looadings and 'v' contains the scores.
function rotate_orthog(u::AbstractMatrix, v::AbstractMatrix)

    # Center the scores
    mn = mean(v, dims = 1)[:]
    for j = 1:size(v, 2)
        v[:, j] .-= mn[j]
    end

    # Handle the one factor case separately.
    if size(u, 2) == 1
        f = norm(u)
        return u ./ f, v .* f, mn
    end

    u1, s1, v1 = svd(v)
    u = u * v1 * diagm(s1)
    v = u1

    # The loading vectors should have unit norm.
    for j = 1:size(u, 2)
        f = norm(u[:, j])
        u[:, j] ./= f
        v[:, j] .*= f
    end

    return u, v, mn
end
