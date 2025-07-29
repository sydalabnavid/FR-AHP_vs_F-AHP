# Fuzzy geometric means for each criterion
geometric_means = [
    (1.4329, 1.8008, 2.1556),
    (1.9288, 2.2112, 2.4874),
    (1.7565, 2.0312, 2.3018),
    (1.3109, 1.6185, 1.9135)
]

# Inverse of total fuzzy sum
inverse = (0.1129, 0.1305, 0.1556)

# Function to multiply two fuzzy triangular numbers
def multiply_fuzzy(fuzzy1, fuzzy2):
    l1, m1, u1 = fuzzy1
    l2, m2, u2 = fuzzy2
    return (l1 * l2, m1 * m2, u1 * u2)

# Compute normalized fuzzy weights
normalized_weights = [multiply_fuzzy(gm, inverse) for gm in geometric_means]

# Output results
for idx, weight in enumerate(normalized_weights, 1):
    print(f"C{idx}: Normalized Weight = {weight}")