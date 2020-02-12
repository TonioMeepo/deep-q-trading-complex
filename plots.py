import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

for folder in ["train", "test"]:
    numWalks = 100

    path = "./walks/" + folder + "/walk"

    finalRewards = []
    finalDay1 = []
    finalDay2 = []

    x = range(numWalks)
    for i in x:
        rewardAcc=0
        rewardAcc1=0
        rewardAcc2=0
        df = pd.read_csv(path+str(i)+".csv",  index_col="episode")
        for index , row in df.iterrows():
            rewardAcc+=row.to_dict()['reward']
            if row.to_dict()['day'] == 1:
                rewardAcc1+=row.to_dict()['reward']
            else:
                rewardAcc2+=row.to_dict()['reward']
        finalRewards.append(rewardAcc)
        finalDay1.append(rewardAcc1)
        finalDay2.append(rewardAcc2)



    plt.figure()
    plt.plot(finalRewards)
    plt.title("Final Rewards")
    plt.savefig("figures/"+folder+"/finalRewards.png")
    plt.show()

    plt.figure()
    plt.plot(finalDay1)
    plt.title("Final Rewards Day 1")
    plt.savefig("figures/"+folder+"/finalRewardsDay1.png")
    plt.show()

    plt.figure()
    plt.plot(finalDay2)
    plt.title("Final Rewards Day 2")
    plt.savefig("figures/"+folder+"/finalRewardsDay2.png")
    plt.show()

"""
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


l1 = train['reward'].tolist()
l2 = train['day'].tolist()

tupleList = list(map(lambda x, y:(x,y), l1, l2))
"""