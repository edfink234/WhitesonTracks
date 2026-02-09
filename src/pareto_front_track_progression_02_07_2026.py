import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your log
depths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
max_sizes = [9, 19, 29, 39, 49, 59, 69, 76, 85, 95, 104, 107, 117, 125, 135, 141, 150, 160]
sse_values = [63574.9, 27980.7, 12477.4, 7773.74, 4674.79, 3120, 2548.01, 1572.9,
              1210.09, 949.357, 807.673, 596.653, 445.276, 343.709, 304.552,
              276.834, 240.955, 187.195]

plt.figure(figsize=(10, 6))

# Plotting the Pareto Front
plt.plot(max_sizes, sse_values, marker='o', linestyle='-', color='#2c3e50', linewidth=2, label='Pareto Front')

# Adding annotations for specific "milestone" depths
for i, d in enumerate(depths):
    if d in [5, 10, 15, 22]:  # Labeling key points to keep it clean
        plt.annotate(f'Depth {d}', (max_sizes[i], sse_values[i]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Log scale for Y-axis often makes SR Pareto plots easier to read
plt.yscale('log')

plt.title('Symbolic Regression Pareto Front: Model Complexity vs. Error', fontsize=14)
plt.xlabel('Model Complexity (Expression Size / Nodes)', fontsize=12)
plt.ylabel('Sum of Squared Errors (Log Scale)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.tight_layout()
plt.show()
