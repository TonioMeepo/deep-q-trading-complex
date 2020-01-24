import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train = pd.read_csv("resultsTrain.csv", index_col="episode")
plt.plot(train['operation'].tolist())
plt.savefig("figures/operation.png")
plt.title("operation")
plt.show()

train = pd.read_csv("resultsTrain.csv", index_col="episode")
plt.plot(train['operation'].tolist())
plt.savefig("figures/operation.png")
plt.title("operation")
plt.show()

x = np.arange(3)

plt.bar(x,height=[train['hold'].tolist()[-1],train['long'].tolist()[-1],train['short'].tolist()[-1]])

plt.xticks(x, ['Hold','Long','Short'])
plt.savefig("figures/percentages.png")
plt.title("Percentages")
plt.show()