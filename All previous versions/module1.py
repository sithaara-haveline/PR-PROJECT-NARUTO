import random
import time
import matplotlib.pyplot as plt

# Merge Sort Function
def merge_sort(arr):

    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2

    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):

    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):

        if left[i] <= right[j]:
            result.append(left[i])
            i += 1

        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


# Different input sizes
sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

times = []

print("Input Size\tExecution Time (seconds)")

for n in sizes:

    arr = [random.randint(1, 1000000) for _ in range(n)]

    start = time.perf_counter()

    merge_sort(arr)

    end = time.perf_counter()

    execution_time = end - start

    times.append(execution_time)

    print(f"{n}\t\t{execution_time:.6f}")

# Plot Graph

plt.figure(figsize=(8,5))
plt.plot(sizes, times, marker='o')
plt.title("Merge Sort Experimental Time Complexity")
plt.xlabel("Input Size")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.show()