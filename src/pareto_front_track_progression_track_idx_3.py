import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your log
max_sizes = [9, 17, 24, 34, 43, 53, 62, 64, 70, 71, 79, 81, 91, 101, 103, 151, 202, 211, 208, 227, 249, 268]
sse_values = [8929.28, 1434, 125.605, 63.3617, 47.1009, 13.0276, 9.09682, 6.02855, 5.13096, 4.91967, 3.75413, 3.28463, 3.21014, 3.1258, 2.75353, 1.04649, 0.33099, 0.325704, 0.212326, 0.135493, 0.104753, 0.101144]

assert(len(max_sizes) == len(sse_values))
              
depths = range(4, 4+len(max_sizes))
print(f"depths = {depths}")
plt.figure(figsize=(10, 6))

# Plotting the Pareto Front
plt.plot(max_sizes, sse_values, marker='o', linestyle='-', color='#2c3e50', linewidth=2, label='Pareto Front')

# Annotations for Old Progression
for i, d in enumerate(depths):
    if d in [4, 10, 17, 23]:
        plt.annotate(f'Depth {d}', (max_sizes[i], sse_values[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#2c3e50')

# Log scale for Y-axis often makes SR Pareto plots easier to read
plt.yscale('log')

plt.title('Symbolic Regression Pareto Front: Model Complexity vs. Error', fontsize=14)
plt.xlabel('Model Complexity (Expression Size / Nodes)', fontsize=12)
plt.ylabel('Sum of Squared Errors (Log Scale)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.tight_layout()
plt.show()
