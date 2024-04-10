import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data into the 'dwi' DataFrame.
# Replace 'path/to/your/data.csv' with the actual path to your dataset.
# For example: dwi = pd.read_csv('/Users/canyonmarshall/Desktop/M2Project_RDD/data.csv')
dwi = pd.read_csv('hansen_dwi.csv')

# Ensure seaborn's visual style is set
sns.set_theme(style="whitegrid")

# Create the histogram plot
plt.figure(figsize=(10, 6))
sns.histplot(dwi['bac1'], binwidth=0.001, color="#8aa1b4", kde=False)

# Add vertical lines at the specified x intercepts
plt.axvline(x=0.08, linewidth=1, linestyle='--', color='tomato', alpha=0.7)
plt.axvline(x=0.15, linewidth=1, linestyle='--', color='tomato', alpha=0.7)

# Set labels and title
plt.xlabel('BAC', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('BAC Histogram\nFigure 1, Hansen (2015)', fontsize=15)

# Adjust y-axis limits for additional space at the top
plt.ylim(0, plt.ylim()[1] * 1.1)

# Customize tick label font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set background color
plt.gca().set_facecolor('white')
plt.grid(True)

# Display the plot
plt.show()
