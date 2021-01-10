import numpy as np
import matplotlib.pyplot as plt

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
plt.plot(x1, y1, 'o--', color=color1)

plt.plot(x1, y2, 'o--', color=color2)
plt.title('Performance Complementary')
plt.ylabel('Algorithm Performance')
plt.xlabel('Instance')
plt.xticks(x1)

plt.show()

fig.savefig("plotted/performance_complementary.pdf", bbox_inches='tight')