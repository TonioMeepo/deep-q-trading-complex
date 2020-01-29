import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



train = pd.read_csv("resultsTrain.csv", index_col="episode")

folder='training'

plt.figure()
plt.plot(train['operation'].tolist())
plt.title("OPERATION")
plt.savefig("figures/"+folder+"/operation.png")
plt.show()

plt.figure()
plt.plot(train['reward'].tolist())
plt.title("REWARD")
plt.axhline(0,color="grey")
plt.grid(axis='y')
plt.savefig("figures/"+folder+"/reward.png")
plt.show()


plt.figure()
x = np.arange(3)
plt.bar(x,height=[train['hold'].tolist()[-1],train['long'].tolist()[-1],train['short'].tolist()[-1]])
plt.xticks(x, ['Hold','Long','Short'])
plt.title("PERCENTAGES")
plt.grid(axis='y')
plt.savefig("figures/"+folder+"/percentages.png")
plt.show()



plt.figure()
plt.plot(train['capital'].tolist())
plt.title("CAPITAL")
plt.axhline(0,color="grey")
plt.savefig("figures/"+folder+"/capital.png")
plt.show()