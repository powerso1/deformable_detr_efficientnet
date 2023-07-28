import matplotlib.pyplot as plt

# Define the colors
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Create a bar chart of the colors
plt.bar(range(len(COLORS)), [1] * len(COLORS), color=COLORS)

# Set the x-axis tick labels and title
plt.xticks(range(len(COLORS)), ["Color {}".format(i + 1)
           for i in range(len(COLORS))])
plt.xlabel("Colors")
plt.title("Color Bar Chart")

# Show the plot
plt.savefig('color_bar.png')
