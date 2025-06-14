import numpy as np

array = np.random.randint(1, 51, size=(5, 4))

print("Original Array:")
print(array)

anti_diagonal = np.array([array[i, 3 - i] for i in range(min(array.shape))])
print("\nAnti-Diagonal Elements:")
print(anti_diagonal)

max_in_each_row = np.max(array, axis=1)
print("\nMaximum Value in Each Row:")
print(max_in_each_row)

mean_of_array = np.mean(array)
new_array = array[array <= mean_of_array]
print("\nNew Array (elements less than or equal to the mean):")
print(new_array)

def numpy_boundary_traversal(matrix):
    if matrix.size == 0:
        return []

    rows, cols = matrix.shape
    boundary_elements = []

    boundary_elements.extend(matrix[0, :].tolist())

    if rows > 1:
        boundary_elements.extend(matrix[1:, cols - 1].tolist())

    if rows > 1:
        boundary_elements.extend(matrix[rows - 1, :-1][::-1].tolist())

    if cols > 1:
        boundary_elements.extend(matrix[1:-1, 0][::-1].tolist())

    return boundary_elements

boundary_list = numpy_boundary_traversal(array)
print("\nBoundary Traversal:")
print(boundary_list)