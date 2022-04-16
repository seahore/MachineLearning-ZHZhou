from datetime import datetime
import json
from random import randint
import sys
import matplotlib.pyplot as plt

clusters = None
with open(sys.argv[1], encoding='utf-8') as file:
    clusters = json.loads(file.read().rstrip())

plot_data = []
for c in clusters:
    t = ([],[])
    for i in c:
        t[0].append(i[0])
        t[1].append(i[1])
    plot_data.append(t)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig, ax = plt.subplots()
for i in range(len(plot_data)):
    ax.scatter(plot_data[i][0], plot_data[i][1], c=colors[i])
ax.set_xlabel('Density')
ax.set_ylabel('Sweetness')
plt.savefig(f'./{randint(10000000,99999999)}.png')