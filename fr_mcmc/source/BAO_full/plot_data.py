import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from CSV files
bao_full_1 = pd.read_csv('BAO_full_1.csv')
bao_full_2 = pd.read_csv('BAO_full_2.csv')

# Display the first few rows of each dataframe to verify the data
print("BAO_full_1:")
print(bao_full_1.head())
print("\nBAO_full_2:")
print(bao_full_2.head())

# Plot the data
plt.figure(figsize=(10, 6))

# Plot BAO_full_1 data
sns.scatterplot(data=bao_full_1, x='z', y='Dist', label='BAO_full_1', color='blue')

# Plot BAO_full_2 data
sns.scatterplot(data=bao_full_2, x='z', y='Dist', label='BAO_full_2', color='red')

# Add labels and title
plt.xlabel('Redshift (z)')
plt.ylabel('Distance (Dist)')
plt.title('BAO Distance Measurements')
plt.legend()

# Show the plot
plt.show()