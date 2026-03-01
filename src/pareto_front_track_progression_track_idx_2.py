import matplotlib.pyplot as plt
import numpy as np

# Data extracted from your log
depths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
max_sizes = [9, 19, 29, 39, 49, 59, 69, 76, 85, 95, 104, 107, 117, 125, 135, 141, 150, 160]
sse_values = [63574.9, 27980.7, 12477.4, 7773.74, 4674.79, 3120, 2548.01, 1572.9,
              1210.09, 949.357, 807.673, 596.653, 445.276, 343.709, 304.552,
              276.834, 240.955, 187.195]

new_max_sizes = [25, 27, 36, 47, 56, 65, 74, 82, 92, 100, 104, 106, 116, 123, 133, 136, 143, 147,
    156, 165, 165, 170, 179, 188, 197, 206, 215, 223, 233, 243, 251,
    261, 270, 283, 292, 300, 303, 328, 345, 361, 379, 416, 433, 444,
    456, 477, 501, 517, 536, 551, 574, 605, 646, 674, 695, 719, 737, 755]
new_sse_values = [14319.8, 12195.8, 8860.18, 7999.78, 6673.63, 5690.95, 4534.99, 4038.1, 3282.08, 3129.52, 2869.33,
    3066.75, 2784.82, 2597.84, 2315.72, 2146.99, 1981.55, 1876.93,
    1739.82, 1624.11, 1481.92, 1329.49, 1204.7, 1123.13, 1031.75,
    951.286, 855.006, 808.341, 768.245, 730.392, 646.575, 608.954,
    571.92, 280.832, 255.476, 236.706, 217.239, 180.43, 152.268,
    134.685, 120.946, 93.0048, 82.6324, 70.7122, 58.6358, 44.6747,
    37.2366, 33.7009, 26.6581, 22.3081, 19.1926, 16.2664, 12.9048,
    9.766, 8.37384, 7.15936, 6.44699, 6.06383]
new_depths = list(range(6, 6+len(new_sse_values)))

plt.figure(figsize=(10, 6))

# Plotting the Pareto Front
plt.plot(max_sizes, sse_values, marker='o', linestyle='-', color='#2c3e50', linewidth=2, label='Pareto Front Old')
plt.plot(new_max_sizes, new_sse_values, marker='o', linestyle='-', color='#6aa84f', linewidth=2, label='Pareto Front New')

# Annotations for Old Progression
for i, d in enumerate(depths):
    if d in [5, 10, 15, 22]:
        plt.annotate(f'Depth {d}', (max_sizes[i], sse_values[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#2c3e50')

# Annotations for New Progression (Added Milestones)
for i, d in enumerate(new_depths):
    if d in [26, 38, 50, 61]:
        plt.annotate(f'Depth {d}', (new_max_sizes[i], new_sse_values[i]),
                     textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='#38761d', weight='bold')

# Log scale for Y-axis often makes SR Pareto plots easier to read
plt.yscale('log')

plt.title('Symbolic Regression Pareto Front: Model Complexity vs. Error', fontsize=14)
plt.xlabel('Model Complexity (Expression Size / Nodes)', fontsize=12)
plt.ylabel('Sum of Squared Errors (Log Scale)', fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

plt.tight_layout()
plt.show()
