import numpy as np
import matplotlib.pyplot as plt

def input_transformation_matrix():

    print("\n" + "=" * 60)
    print("ENTER 3x3 TRANSFORMATION MATRIX")
    print("=" * 60)

    print("Enter each row as 3 numbers separated by spaces.")
    print("Example:")
    print("1 1 0")
    print("0 2 0")
    print("0 0 3\n")
    matrix_rows = []
    for i in range(3):
        while True:
            try:
                row = input(
                    f"Enter row {i + 1}: "
                ).split()
                if len(row) != 3:
                    print(
                        "ERROR: Each row must contain exactly "
                        "3 numbers."
                    )
                    continue
                row = [float(value) for value in row]
                matrix_rows.append(row)
                break
            except ValueError:
                print(
                    "ERROR: Please enter only numeric values."
                )

    matrix = np.array(
        matrix_rows,
        dtype=float
    )
    return matrix

def read_points_from_file(filename):
    points = []
    with open(filename, "r") as file:
        for line_number, line in enumerate(
            file,
            start=1
        ):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            values = line.split()
            if len(values) != 3:
                raise ValueError(
                    f"Invalid point at line {line_number}. "
                    "Each point must contain exactly 3 values."
                )
            try:
                point = [
                    float(value)
                    for value in values
                ]
            except ValueError:
                raise ValueError(
                    f"Invalid numeric value at "
                    f"line {line_number}."
                )
            points.append(point)
    if len(points) == 0:
        raise ValueError(
            "No points were found in the text file."
        )
    return np.array(
        points,
        dtype=float
    )

def transform_points(matrix, points):

    """
    Applies:

        P' = A P

    to every point.
    """

    transformed_points = (
        matrix @ points.T
    ).T

    return transformed_points

def calculate_eigenbasis(matrix):

    eigenvalues, eigenvectors = np.linalg.eig(
        matrix
    )

    # Remove tiny numerical imaginary parts
    eigenvalues = np.real_if_close(
        eigenvalues
    )

    eigenvectors = np.real_if_close(
        eigenvectors
    )

    return eigenvalues, eigenvectors

def convert_to_eigenbasis(
        points,
        eigenvectors
):

    """
    If

        P = V C

    then

        C = V^-1 P

    where V contains the eigenvectors
    as its columns.
    """

    determinant = np.linalg.det(
        eigenvectors
    )

    if abs(determinant) < 1e-10:

        raise ValueError(
            "The matrix does not have 3 linearly "
            "independent eigenvectors.\n"
            "Therefore, a complete eigenbasis "
            "cannot be constructed."
        )

    V_inverse = np.linalg.inv(
        eigenvectors
    )

    eigen_coordinates = (
        V_inverse @ points.T
    ).T

    return eigen_coordinates

def set_equal_axes(
        ax,
        all_points
):

    all_points = np.asarray(
        all_points
    )

    x_min = np.min(
        all_points[:, 0]
    )

    x_max = np.max(
        all_points[:, 0]
    )

    y_min = np.min(
        all_points[:, 1]
    )

    y_max = np.max(
        all_points[:, 1]
    )

    z_min = np.min(
        all_points[:, 2]
    )

    z_max = np.max(
        all_points[:, 2]
    )

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    max_range = max(
        x_range,
        y_range,
        z_range
    )

    if max_range == 0:
        max_range = 1

    x_middle = (
        x_max + x_min
    ) / 2

    y_middle = (
        y_max + y_min
    ) / 2

    z_middle = (
        z_max + z_min
    ) / 2

    ax.set_xlim(
        x_middle - max_range / 2,
        x_middle + max_range / 2
    )

    ax.set_ylim(
        y_middle - max_range / 2,
        y_middle + max_range / 2
    )

    ax.set_zlim(
        z_middle - max_range / 2,
        z_middle + max_range / 2
    )

def plot_standard_basis(ax):

    axis_length = 1
    ax.quiver(
        0, 0, 0,
        axis_length, 0, 0,
        color="red",
        arrow_length_ratio=0.1,
        linewidth=2
    )
    ax.quiver(
        0, 0, 0,
        0, axis_length, 0,
        color="green",
        arrow_length_ratio=0.1,
        linewidth=2
    )
    ax.quiver(
        0, 0, 0,
        0, 0, axis_length,
        color="blue",
        arrow_length_ratio=0.1,
        linewidth=2
    )
    ax.text(
        axis_length,
        0,
        0,
        "X"
    )
    ax.text(
        0,
        axis_length,
        0,
        "Y"
    )
    ax.text(
        0,
        0,
        axis_length,
        "Z"
    )

def plot_eigenbasis(
        ax,
        eigenvectors
):
    colors = [
        "purple",
        "orange",
        "brown"
    ]
    labels = [
        "Eigenvector 1",
        "Eigenvector 2",
        "Eigenvector 3"
    ]
    for i in range(3):
        vector = eigenvectors[:, i]
        norm = np.linalg.norm(
            vector
        )
        if norm > 1e-10:
            vector = vector / norm
        ax.quiver(
            0,
            0,
            0,
            vector[0],
            vector[1],
            vector[2],
            color=colors[i],
            linewidth=2,
            arrow_length_ratio=0.12
        )
        ax.text(
            vector[0],
            vector[1],
            vector[2],
            labels[i],
            color=colors[i]
        )

def plot_points(
        ax,
        points,
        label,
        marker
):

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        marker=marker,
        s=60,
        label=label
    )

def visualize_standard_basis(
        original_points,
        transformed_points,
        eigenvectors
):

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    plot_points(
        ax,
        original_points,
        "Original Points",
        "o"
    )

    plot_points(
        ax,
        transformed_points,
        "Transformed Points",
        "^"
    )
    for original, transformed in zip(
        original_points,
        transformed_points
    ):

        ax.plot(
            [
                original[0],
                transformed[0]
            ],
            [
                original[1],
                transformed[1]
            ],
            [
                original[2],
                transformed[2]
            ],
            linestyle="--",
            alpha=0.4
        )
    plot_standard_basis(ax)
    plot_eigenbasis(
        ax,
        eigenvectors
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_zlabel(
        "Z"
    )

    ax.set_title(
        "3D Transformation - Normal Basis"
    )

    ax.legend()

    all_points = np.vstack(
        (
            original_points,
            transformed_points,
            np.zeros((1, 3)),
            eigenvectors.T
        )
    )

    set_equal_axes(
        ax,
        all_points
    )

    plt.tight_layout()

    plt.show()

def visualize_eigenbasis(
        eigen_coordinates,
        eigenvalues
):

    transformed_eigen_coordinates = (
        eigen_coordinates * eigenvalues
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Original points in eigenbasis
    plot_points(
        ax,
        eigen_coordinates,
        "Original Points",
        "o"
    )
    plot_points(
        ax,
        transformed_eigen_coordinates,
        "Transformed Points",
        "^"
    )
    for original, transformed in zip(
        eigen_coordinates,
        transformed_eigen_coordinates
    ):

        ax.plot(
            [
                original[0],
                transformed[0]
            ],
            [
                original[1],
                transformed[1]
            ],
            [
                original[2],
                transformed[2]
            ],
            linestyle="--",
            alpha=0.4
        )
    all_points = np.vstack(
        (
            eigen_coordinates,
            transformed_eigen_coordinates,
            np.zeros((1, 3))
        )
    )

    max_value = np.max(
        np.abs(all_points)
    )

    if max_value == 0:
        max_value = 1
    axis_length = max_value * 1.2
    ax.quiver(
        0, 0, 0,
        axis_length, 0, 0,
        color="purple",
        linewidth=2,
        arrow_length_ratio=0.08
    )

    ax.quiver(
        0, 0, 0,
        0, axis_length, 0,
        color="orange",
        linewidth=2,
        arrow_length_ratio=0.08
    )

    ax.quiver(
        0, 0, 0,
        0, 0, axis_length,
        color="brown",
        linewidth=2,
        arrow_length_ratio=0.08
    )

    ax.text(
        axis_length,
        0,
        0,
        "Eigenbasis e₁"
    )

    ax.text(
        0,
        axis_length,
        0,
        "Eigenbasis e₂"
    )

    ax.text(
        0,
        0,
        axis_length,
        "Eigenbasis e₃"
    )

    ax.set_xlabel(
        "Eigen-coordinate 1"
    )

    ax.set_ylabel(
        "Eigen-coordinate 2"
    )

    ax.set_zlabel(
        "Eigen-coordinate 3"
    )

    ax.set_title(
        "3D Transformation - Eigenbasis"
    )

    ax.legend()

    set_equal_axes(
        ax,
        all_points
    )

    plt.tight_layout()

    plt.show()

def print_results(
        matrix,
        points,
        transformed_points,
        eigenvalues,
        eigenvectors,
        eigen_coordinates
):

    np.set_printoptions(
        precision=4,
        suppress=True
    )

    print("\n" + "=" * 60)
    print("TRANSFORMATION MATRIX")
    print("=" * 60)

    print(matrix)

    print("\n" + "=" * 60)
    print("ORIGINAL POINTS")
    print("=" * 60)

    print(points)

    print("\n" + "=" * 60)
    print("TRANSFORMED POINTS")
    print("=" * 60)

    print(transformed_points)

    print("\n" + "=" * 60)
    print("EIGENVALUES")
    print("=" * 60)

    for i, value in enumerate(
        eigenvalues
    ):

        print(
            f"λ{i + 1} = {value:.4f}"
        )

    print("\n" + "=" * 60)
    print("EIGENVECTORS")
    print("=" * 60)

    for i in range(3):

        print(
            f"\nEigenvector {i + 1}:"
        )

        print(
            eigenvectors[:, i]
        )

    print("\n" + "=" * 60)
    print("EIGENVECTOR MATRIX V")
    print("=" * 60)

    print(eigenvectors)

    print("\n" + "=" * 60)
    print("POINTS IN EIGENBASIS")
    print("=" * 60)

    print(eigen_coordinates)

    print("\n" + "=" * 60)
    print("DIAGONAL MATRIX D")
    print("=" * 60)

    D = np.diag(
        eigenvalues
    )

    print(D)

    print("\n" + "=" * 60)
    print("VERIFICATION: A = V D V^-1")
    print("=" * 60)

    V = eigenvectors

    V_inverse = np.linalg.inv(V)

    reconstructed_matrix = (
        V @ D @ V_inverse
    )

    print("\nOriginal matrix A:")
    print(matrix)

    print("\nReconstructed V D V^-1:")
    print(reconstructed_matrix)

    print("\nDifference:")
    print(
        matrix - reconstructed_matrix
    )

def main():

    print("\n")
    print("=" * 60)
    print("3D LINEAR TRANSFORMATION VISUALIZER")
    print("=" * 60)

    matrix = input_transformation_matrix()

    print("\n" + "=" * 60)
    print("POINT INPUT")
    print("=" * 60)

    filename = input(
        "Enter points file name "
        "(press Enter for points.txt): "
    ).strip()

    if filename == "":
        filename = "points.txt"

    try:

        points = read_points_from_file(
            filename
        )

    except FileNotFoundError:

        print(
            f"\nERROR: Could not find '{filename}'."
        )

        print(
            "Make sure the file is in the same "
            "folder as this Python program."
        )

        return

    except ValueError as error:

        print(
            f"\nERROR: {error}"
        )

        return

    print(
        f"\nSuccessfully loaded "
        f"{len(points)} 3D points."
    )

    transformed_points = transform_points(
        matrix,
        points
    )

    try:

        eigenvalues, eigenvectors = (
            calculate_eigenbasis(matrix)
        )

        if np.iscomplexobj(
            eigenvalues
        ):

            print(
                "\nERROR:"
            )

            print(
                "The transformation matrix has "
                "complex eigenvalues."
            )

            print(
                "A real 3D eigenbasis cannot be "
                "visualized by this program."
            )

            return

        if np.iscomplexobj(
            eigenvectors
        ):

            print(
                "\nERROR:"
            )

            print(
                "The transformation matrix has "
                "complex eigenvectors."
            )

            return

        eigenvalues = np.asarray(
            eigenvalues,
            dtype=float
        )

        eigenvectors = np.asarray(
            eigenvectors,
            dtype=float
        )

    except np.linalg.LinAlgError:

        print(
            "\nERROR: Could not calculate "
            "eigenvalues/eigenvectors."
        )

        return

    try:

        eigen_coordinates = (
            convert_to_eigenbasis(
                points,
                eigenvectors
            )
        )

    except ValueError as error:

        print(
            f"\nERROR: {error}"
        )

        print(
            "\nThe normal-basis transformation "
            "can still be displayed."
        )

        # Display normal basis visualization
        visualize_standard_basis(
            points,
            transformed_points,
            eigenvectors
        )

        return

    print_results(
        matrix,
        points,
        transformed_points,
        eigenvalues,
        eigenvectors,
        eigen_coordinates
    )

    print(
        "\nOpening normal-basis visualization..."
    )

    visualize_standard_basis(
        points,
        transformed_points,
        eigenvectors
    )

    print(
        "\nOpening eigenbasis visualization..."
    )

    visualize_eigenbasis(
        eigen_coordinates,
        eigenvalues
    )

if __name__ == "__main__":
    main()