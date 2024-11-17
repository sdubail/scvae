import numpy as np
from scipy.sparse import csr_matrix

# this is a  personal try. original synthetic data generation is handled in the loaders.py


def generate_test_dataset(
    n_examples=1000,
    n_features=100,
    n_classes=5,
    sparsity=0.8,  # Target sparsity level (proportion of zeros)
    n_housekeeping=10,  # Number of housekeeping genes expressed across all classes
    correlation_groups=3,  # Number of correlated gene groups per class
    mean_library_size=20000,  # Average number of total counts per cell
    library_size_std=5000,  # Standard deviation for library size variation
):
    """Generate synthetic RNA-seq data with realistic characteristics."""
    np.random.seed(42)  # For reproducibility

    # 1. Create base expression profiles for each class (same as before)
    class_profiles = []
    genes_per_group = (n_features - n_housekeeping) // (n_classes * correlation_groups)

    # Housekeeping genes profile (same as before)
    housekeeping_profile = np.random.negative_binomial(n=20, p=0.3, size=n_housekeeping)

    for i in range(n_classes):
        profile = np.zeros(n_features)
        profile[:n_housekeeping] = housekeeping_profile

        for group in range(correlation_groups):
            start_idx = (
                n_housekeeping + (i * correlation_groups + group) * genes_per_group
            )
            end_idx = start_idx + genes_per_group

            base_expression = np.random.negative_binomial(
                n=np.random.randint(5, 15), p=np.random.uniform(0.2, 0.4), size=1
            )[0]

            group_profile = np.random.negative_binomial(
                n=base_expression, p=0.3, size=genes_per_group
            )

            profile[start_idx:end_idx] = group_profile

        class_profiles.append(profile)

    # 2. Generate examples with controlled sparsity
    values = []
    labels = []
    library_sizes = np.random.normal(
        mean_library_size, library_size_std, n_examples
    ).astype(int)
    library_sizes = np.maximum(library_sizes, 1000)

    for i in range(n_examples):
        # Pick a class
        class_idx = i % n_classes
        profile = class_profiles[class_idx].copy()

        # Add biological noise
        gene_noise = np.random.negative_binomial(n=2, p=0.5, size=n_features)
        profile = profile + gene_noise

        # Calculate dynamic dropout probabilities to achieve target sparsity
        # Higher expression -> lower dropout probability
        expression_probs = 1 / (1 + np.exp(-profile / profile.mean()))
        dropout_probs = np.clip(
            sparsity + (1 - sparsity) * (1 - expression_probs),
            0.1,  # Minimum dropout probability
            0.99,  # Maximum dropout probability
        )

        # Special handling for housekeeping genes - lower dropout probability
        dropout_probs[:n_housekeeping] *= 0.3

        # Apply dropouts
        dropout_mask = np.random.binomial(1, 1 - dropout_probs)
        profile = profile * dropout_mask

        # Scale to target library size
        if profile.sum() > 0:
            profile = profile * (library_sizes[i] / profile.sum())
            profile = np.round(profile).astype(int)

        values.append(profile)
        labels.append(f"Class_{class_idx}")

    # Convert to sparse matrix
    values = csr_matrix(np.array(values))
    labels = np.array(labels)

    # Verify sparsity level
    achieved_sparsity = (values == 0).sum() / values.size
    print(f"Target sparsity: {sparsity:.3f}")
    print(f"Achieved sparsity: {achieved_sparsity:.3f}")

    # Generate names
    example_names = np.array([f"cell_{i}" for i in range(n_examples)])
    feature_names = np.array([f"gene_{i}" for i in range(n_features)])

    mean_count = values.sum(axis=1).mean()
    # Print summary statistics
    print(f"Data matrix shape: {values.shape}")
    print(f"Number of non-zero entries: {values.getnnz()}")
    print(f"Mean counts per cell: {mean_count:.1f}")
    print(f"Number of classes: {len(np.unique(labels))}")

    # Check if data ok with negative binomial params
    # Theoretical ranges with our parameters
    r_range = [np.exp(-2), np.exp(8)]  # [0.14, 2980]
    p_range = [1e-4, 0.9999]

    # Calculate possible means these ranges can handle
    min_mean = r_range[0] * (1 - p_range[1]) / p_range[1]  # Smallest mean
    max_mean = r_range[1] * (1 - p_range[0]) / p_range[0]  # Largest mean

    print(f"Parameter ranges can handle means from {min_mean:.1f} to {max_mean:.1f}")
    print(f"Your mean count per cell: {mean_count}")
    print(f"Estimated mean count per gene: {mean_count/n_features:.1f}")

    # Check if ranges are appropriate
    if min_mean > mean_count / n_features:
        print("WARNING: Minimum mean too high!")
    if max_mean < mean_count:
        print("WARNING: Maximum mean too low!")

    return {
        "values": values,
        "labels": labels,
        "example names": example_names,
        "feature names": feature_names,
    }
