#import IntradayPolicy
import SpEnv
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import datetime
import telegram as te
from telegramSettings import telegramToken, telegramChatID

bot = te.Bot(token=telegramToken)



epochs = 200
windowLength=1


#842063
trainEnv = SpEnv.SpEnv(maxLimit = 1893600, verbose=True,operationCost=0,observationWindow=600, output='walks/train/walk0.csv')
testEnv = SpEnv.SpEnv(minLimit = 1893601, verbose=True,operationCost=0,observationWindow=600, output='walks/test/walk0.csv')
nb_actions = trainEnv.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(windowLength,) + trainEnv.observation_space.shape))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy(eps = 0.5)



memory = SequentialMemory(limit=10000, window_length=windowLength)



dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    enable_dueling_network=False,
    enable_double_dqn=False,
    memory=memory,
    policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.load_weights("Q.weights")


bot.send_message(chat_id=telegramChatID, text="Experiment started - "+datetime.datetime.now().strftime("%H:%M"))


percIncrement = 100/epochs
perc = 0
for i in range(epochs):
    policy.eps=1
    dqn.fit(trainEnv, nb_steps=10000, visualize=False, verbose=0)
    dqn.test(testEnv, nb_episodes=20, verbose=0, visualize=False)
    perc+=percIncrement
    bot.send_message(chat_id=telegramChatID, text=str(perc)+" % - "+datetime.datetime.now().strftime("%H:%M"))
    trainEnv.changeOutput("walks/train/walk"+str(i+1)+".csv")
    testEnv.changeOutput("walks/test/walk"+str(i+1)+".csv")



dqn.save_weights("Q.weights", overwrite=True)


bot.send_message(chat_id=telegramChatID, text="Experiment ended - "+datetime.datetime.now().strftime("%H:%M"))
