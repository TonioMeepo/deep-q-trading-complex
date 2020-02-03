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

#842063
environment = SpEnv.SpEnv(maxLimit = 1893600, verbose=True,operationCost=0,observationWindow=600, output='resultsTrain.csv')
testEnv = SpEnv.SpEnv(minLimit = 1893601, verbose=True,operationCost=0,observationWindow=600, output='resultsTest.csv')
nb_actions = environment.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(20,) + environment.observation_space.shape))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy(eps = 0.5)



memory = SequentialMemory(limit=100000, window_length=20)
dqn = DQNAgent(model=model, nb_actions=nb_actions,enable_dueling_network=True, memory=memory, nb_steps_warmup=4000,
target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

#dqn.load_weights("Q.weights")

print(datetime.datetime.now())

startingTime=datetime.datetime.now()
bot.send_message(chat_id=telegramChatID, text="Experiment started - "+str(datetime.datetime.now()))

policy.eps=1
dqn.fit(environment, nb_steps=100000, visualize=False, verbose=0)
dqn.save_weights("Q.weights", overwrite=True)

bot.send_message(chat_id=telegramChatID, text="20 %")

policy.eps= 0.5
dqn.fit(environment, nb_steps=100000, visualize=False, verbose=0)
dqn.save_weights("Q.weights", overwrite=True)


bot.send_message(chat_id=telegramChatID, text="40 %")

policy.eps=0.25
dqn.fit(environment, nb_steps=100000, visualize=False, verbose=0)
dqn.save_weights("Q.weights", overwrite=True)


bot.send_message(chat_id=telegramChatID, text="60 %")

policy.eps=0.125
dqn.fit(environment, nb_steps=100000, visualize=False, verbose=0)
dqn.save_weights("Q.weights", overwrite=True)


bot.send_message(chat_id=telegramChatID, text="80 %")

policy.eps=0.0625
dqn.fit(environment, nb_steps=100000, visualize=False, verbose=0)
dqn.save_weights("Q.weights", overwrite=True)


bot.send_message(chat_id=telegramChatID, text="100 %")

bot.send_message(chat_id=telegramChatID, text="Training ended - "+str(datetime.datetime.now()))
print("End of traning")
print(datetime.datetime.now())
dqn.test(testEnv, nb_episodes=2000, verbose=0, visualize=False)

bot.send_message(chat_id=telegramChatID, text="Test ended - "+str(datetime.datetime.now()))
