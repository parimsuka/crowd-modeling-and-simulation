import numpy as np
import matplotlib.pyplot as plt

def read_file_columns(filename="average_values.txt"):
    data = np.loadtxt(filename, delimiter="\t")
    columns = data.T.tolist()
    return columns

columns = read_file_columns()

# Define the infectionRate and recoveryRate values
infectionRate = [0.03, 0.065, 0.1]
recoveryRate = [0.03, 0.065, 0.1]

susceptible_average = columns[0]
infected_average = columns[1]
recovered_average = columns[2]

# Create an array for the x-axis ticks
ticks = np.arange(len(infectionRate) * len(recoveryRate))

# Define the width of each bar
bar_width = 0.25

# Create the grouped bar plot
plt.bar(ticks, susceptible_average, width=bar_width, label='Susceptible')
plt.bar(ticks + bar_width, infected_average, width=bar_width, label='Infected')
plt.bar(ticks + 2 * bar_width, recovered_average, width=bar_width, label='Recovered')

# Set the x-axis labels
plt.xticks(ticks + bar_width, [f"Infection {i}\nRecovery {j}" for i in infectionRate for j in recoveryRate])

# Set the y-axis label
plt.ylabel('Average Number of People')

# Add a legend
plt.legend(loc='upper left')

# Display the plot
plt.show()


