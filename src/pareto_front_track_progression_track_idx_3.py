import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your log
max_sizes = [9, 17, 24, 34, 43, 53, 62, 64, 70, 71, 79, 81, 91, 101, 103, 151, 202, 211, 208, 227, 249, 261, 285, 299, 315, 330, 355, 373, 395, 437, 453, 469]
sse_values = [8929.28, 1434, 125.605, 63.3617, 47.1009, 13.0276, 9.09682, 6.02855, 5.13096, 4.91967, 3.75413, 3.28463, 3.21014, 3.1258, 2.75353, 1.04649, 0.33099, 0.325704, 0.212326, 0.135493, 0.104753, 0.0990222, 0.0988432, 0.0647427, 0.0394207, 0.0353318, 0.0214888, 0.0146172, 0.00951738, 0.00603557, 0.0039912, 0.00323335]

#print(*zip(max_sizes, sse_values))

assert len(max_sizes) == len(sse_values), f"len(max_sizes) = {len(max_sizes)} and len(sse_values) = {len(sse_values)}"
              
depths = list(range(4, 4+len(max_sizes)))
print(f"depths = {depths}")
depths = depths[:-2]
depths.insert(depths.index(22), 22)
depths.insert(depths.index(26), 26)
print(f"depths = {depths}");
plt.figure(figsize=(10, 6))

# Plotting the Pareto Front
plt.plot(max_sizes, sse_values, marker='o', linestyle='-', color='#2c3e50', linewidth=2, label='Pareto Front')

# Annotations for Old Progression
for i, d in enumerate(depths):
    if d in [4, 18, 24, 29, 32]:
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
