import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color1 = '#264653'
color2 = '#2a9d8f'
color3 = '#e76f51'
color4 = '#e9c46a'
color5 = '#251314'

np.random.seed(123)

algorithm1 = np.random.rand(10)
algorithm2 = np.random.rand(10)


x1 = range(1, len(algorithm1) + 1)

y1 = algorithm1
y2 = algorithm2

fig = plt.figure(1, figsize=(9, 3))

plt.subplot(1, 1, 1)
for x, y in zip(x1, y1):
    plt.bar(x-0.15, y, color=color1, width=0.3, zorder=6)
for x, y in zip(x1, y2):
    plt.bar(x+0.15, y, color=color2, width=0.3, zorder=6)
plt.title('Performance Complementary')
plt.ylabel('Solver Performance')
plt.xlabel('Instance')
plt.xticks(x1)

plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

l1 = mpatches.Patch(color=color1, label="Solver A")
l2 = mpatches.Patch(color=color2, label="Solver B")


plt.legend(handles=[l1, l2], loc=2)

plt.show()

fig.savefig("plotted/performance_complementary.pdf", bbox_inches='tight')