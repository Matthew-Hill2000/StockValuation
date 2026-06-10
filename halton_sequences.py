import matplotlib.pyplot as plt
import numpy as np


def halton_sequence(index, base):
    """
    Compute the Halton sequence value for a given index and base.

    Parameters:
    - index: The index in the Halton sequence (1-based index).
    - base: The base for the Halton sequence.

    Returns:
    - value: The Halton sequence value for the given index and base.
    """
    fractional_part = 0
    factor = 1
    while index > 0:
        factor /= base
        fractional_part += factor * (index % base)
        index = index // base
    return fractional_part


def generate_halton_vectors(base_a, base_b, num_vectors):
    """
    Generate a matrix of Halton sequence values with two different bases.

    Parameters:
    - base_a: The base for the first Halton sequence dimension.
    - base_b: The base for the second Halton sequence dimension.
    - num_vectors: The number of Halton vectors to generate.

    Returns:
    - halton_matrix: A 2D numpy array where each row contains subsequent Halton sequence values for the specified bases.
    """
    halton_matrix = np.zeros((num_vectors, 2))

    for i in range(num_vectors):
        halton_matrix[i, 0] = halton_sequence(i + 1, base_a)
        halton_matrix[i, 1] = halton_sequence(i + 1, base_b)

    return halton_matrix


def apply_box_muller_transform(halton_matrix):
    """
    Apply the Box-Muller transform to a matrix of Halton sequence values.

    Parameters:
    - halton_matrix: A 2D numpy array where each row contains two Halton sequence values (in [0, 1)).

    Returns:
    - normal_random_vars: A 1D numpy array containing normally distributed random variables.
    """
    x1 = halton_matrix[:, 0]
    x2 = halton_matrix[:, 1]

    # Apply the Box-Muller transform
    transformed_1 = np.cos(2 * np.pi * x2) * np.sqrt(-2 * np.log(x1))
    transformed_2 = np.sin(2 * np.pi * x1) * np.sqrt(-2 * np.log(x2))

    # Combine transformed variables into a single array
    normal_random_vars = np.empty((x1.size * 2), dtype=x1.dtype)
    normal_random_vars[0::2] = transformed_1
    normal_random_vars[1::2] = transformed_2

    return normal_random_vars


def main():
    num_points = 1000
    base_a = 2
    base_b = 3

    # Generate Halton sequence vectors
    halton_vectors = generate_halton_vectors(base_a, base_b, num_points)

    # Apply Box-Muller transform
    normal_vars = apply_box_muller_transform(halton_vectors)

    # Plot the Halton sequence points
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(halton_vectors[:, 0], halton_vectors[:, 1], alpha=0.5, edgecolor="k")
    plt.title("Halton Sequence Points")
    plt.xlabel("Halton Dimension 1")
    plt.ylabel("Halton Dimension 2")
    plt.grid(True)

    # Plot the normal distribution
    plt.subplot(1, 2, 2)
    plt.hist(
        normal_vars,
        bins=int(30 + np.log(num_points / 100) * 10),
        density=True,
        alpha=0.6,
        color="g",
    )
    mu, std = np.mean(normal_vars), np.std(normal_vars)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)
    plt.plot(x, p, "k", linewidth=2)
    plt.title("Histogram of Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
