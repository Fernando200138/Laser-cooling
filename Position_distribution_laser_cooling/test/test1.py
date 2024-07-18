import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a figure
fig = plt.figure(figsize=(10, 6))

# Define the grid layout with 2 rows and 2 columns
gs = GridSpec(2, 2, width_ratios=[1, 2])

# Add the first text to the first row of the first column
ax_text1 = fig.add_subplot(gs[0, 0])
ax_text1.axis('off')  # Turn off the axis
text1 = "First text aligned with the first plot."
ax_text1.text(0.5, 0.5, text1, va='center', ha='center', fontsize=12)

# Add the second text to the second row of the first column
ax_text2 = fig.add_subplot(gs[1, 0])
ax_text2.axis('off')  # Turn off the axis
text2 = "Second text aligned with the second plot."
ax_text2.text(0.5, 0.5, text2, va='center', ha='center', fontsize=12)

# Add the first graph to the second column, first row
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot([0, 1, 2, 3], [0, 1, 4, 9])  # Example plot
ax1.set_title('First Graph')

# Add the second graph to the second column, second row
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot([0, 1, 2, 3], [0, 1, 2, 3])  # Example plot
ax2.set_title('Second Graph')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()

