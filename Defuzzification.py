def defuzzify_and_normalize_fuzzy_weights(fuzzy_weights):
    # Step 5: Defuzzify using centroid method
    defuzzified = [sum(triple) / 3 for triple in fuzzy_weights]
    
    # Step 6: Normalize the defuzzified weights
    total = sum(defuzzified)
    normalized = [round(w / total, 4) for w in defuzzified]
    
    return defuzzified, total, normalized

# Example fuzzy weights (from your normalized fuzzy weights step)
fuzzy_weights = [
    (0.1618, 0.2350, 0.3353),
    (0.2177, 0.2886, 0.3869),
    (0.1983, 0.2651, 0.3580),
    (0.1480, 0.2112, 0.2976)
]

# Run the function
defuzzified_weights, total_weight, normalized_weights = defuzzify_and_normalize_fuzzy_weights(fuzzy_weights)

# Display the results
print("Defuzzified Weights (G_i):", defuzzified_weights)
print("Total Defuzzified Weight: ", total_weight)
print("Normalized Weights (H_i):", normalized_weights)