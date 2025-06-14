import numpy as np

array = np.random.uniform(0, 10, 20)

print("Original Array:")
print(array)

rounded_array = np.round(array, 2)
print("\nRounded Array (2 decimal places):")
print(rounded_array)

minimum = np.min(array)
maximum = np.max(array)
median = np.median(array)

print("\nMinimum:", minimum)
print("Maximum:", maximum)
print("Median:", median)

array[array < 5] = array[array < 5] ** 2
print("\nArray with elements less than 5 replaced by their squares:")
print(array)

def numpy_alternate_sort(array):
    sorted_array = np.sort(array)
    result = np.zeros_like(sorted_array)
    left = 0
    right = len(sorted_array) - 1
    for i in range(len(sorted_array)):
        if i % 2 == 0:
            result[i] = sorted_array[left]
            left += 1
        else:
            result[i] = sorted_array[right]
            right -= 1
    return result

alternating_sorted_array = numpy_alternate_sort(array.copy())
print("\nAlternating Sorted Array:")
print(alternating_sorted_array)