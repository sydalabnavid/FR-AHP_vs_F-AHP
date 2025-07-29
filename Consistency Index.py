import numpy as np

def defuzzify(triangular_number):
    l, m, u = triangular_number
    return (l + m + u) / 3

def calculate_cr(fuzzy_matrix, ri_value):
    # Step 1: Defuzzify the fuzzy matrix
    defuzzified_matrix = np.array([[defuzzify(cell) for cell in row] for row in fuzzy_matrix])

    # Step 2: Normalize the matrix column-wise
    column_sums = defuzzified_matrix.sum(axis=0)
    normalized_matrix = defuzzified_matrix / column_sums

    # Step 3: Compute the priority vector (average of rows)
    priority_vector = normalized_matrix.mean(axis=1)

    # Step 4: Weighted sum vector
    weighted_sum_vector = defuzzified_matrix @ priority_vector

    # Step 5: Calculate lambda_max
    lambda_max = (weighted_sum_vector / priority_vector).mean()

    # Step 6: CI and CR
    n = len(fuzzy_matrix)
    ci = (lambda_max - n) / (n - 1)
    cr = ci / ri_value

    return lambda_max, ci, cr

if __name__ == "__main__":
    # Example matrix: 4x4 fuzzy pairwise matrix
    fuzzy_matrix = [
        [(1,1,1), (2,3,4), (4,5,6), (4,5,6)],
        [(0.25,0.33,0.5), (1,1,1), (1,1,1), (2,3,4)],
        [(0.1667,0.2,0.25), (1,1,1), (1,1,1), (4,5,6)],
        [(0.1667,0.2,0.25), (0.25,0.33,0.5), (0.1667,0.2,0.25), (1,1,1)]
    ]

    ri_value = 0.90  # For N = 4

    lambda_max, ci, cr = calculate_cr(fuzzy_matrix, ri_value)

    print(f"Lambda max (\lambda_max): {lambda_max:.4f}")
    print(f"Consistency Index (CI): {ci:.4f}")
    print(f"Consistency Ratio (CR): {cr:.4f}")
\end{lstlisting}