from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import model_from_json
# import tensorflow as tf                                                               
import random
import numpy as np
import os.path

class Agent:
    def __init__(self):

        self.tensorShape = (76, 384, 4)
        # self.tensorShape = (45, 230, 4)
        #This is the actual Neural net
        model = Sequential([ 
            Conv2D(32, (8,8), input_shape=self.tensorShape,
                   strides=(2,2), activation='relu'),
            MaxPooling2D(pool_size=(5,5), strides=(2, 2)),
            Conv2D(64, (4,4), activation='relu', strides=(1,1)),
            MaxPooling2D(pool_size=(7,7), strides=(3,3)),
            Conv2D(128, (1, 1), strides=(1,1), activation='relu'),
            MaxPooling2D(pool_size=(3,3), strides=(3,3)),
            Flatten(),
            Dense(self.tensorShape[1], activation='relu'),
            Dense(64, activation="relu", name="layer1"),
            Dense(8, activation="relu", name="layer2"),
            Dense(3, activation="linear", name="layer3"),
        ])
        #pick your learning rate here
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001)) 
        #This is where you import your pretrained weights
        if os.path.isfile("DinoGameSpeed4.h5"):
            model.load_weights("DinoGameSpeed4.h5")
        self.model = model
        self.memory = []
        # Print the model summary if you want to see what it looks like
        # print(self.model.summary()) 
        self.xTrain = []
        self.yTrain = []
        self.loss = []
        self.location = 0


    def predict(self, state):
        stateConv = state
        qval = self.model.predict(np.reshape(stateConv, (1, *self.tensorShape)))
        return qval

    def act(self, state):
        qval = self.predict(state)
        #you can either pick softmax or epislon greedy actions.
        #To pick Softmax, un comment the bottom 2 lines and delete everything below that 
        # prob = tf.nn.softmax(tf.math.divide((qval.flatten()), 1)) 
        # action = np.random.choice(range(3), p=np.array(prob))

        #Epsilon-Greedy actions->
        z = np.random.random()
        epsilon = 0.004
        if self.location > 1000:
            epsilon = 0.05
        # epsilon = 0
        if z > epsilon:
            return np.argmax(qval.flatten())
        else:
            return np.random.choice(range(3))
        # return action

    # This function stores experiences in the experience replay
    def remember(self, state, nextState, action, reward, done, location):
        self.location = location
        self.memory.append(np.array(state.flatten().tolist() + nextState.flatten().tolist() + [action, reward, done]))
        # self.memory.append([state, nextState, action, reward, done])

    #This is where the AI learns
    def learn(self):
        #Feel free to tweak this. This number is the number of experiences the AI learns from every round
        self.batchSize = 512 

        #If you don't trim the memory, your GPU might run out of memory during training. 
        #I found 35000 works well
        if len(self.memory) > 35000:
            self.memory = []
            print("trimming memory")
        if len(self.memory) < self.batchSize:
            print("too little info")
            return  
        batch = random.sample(self.memory, self.batchSize)

        self.learnBatch(batch)

    #The alpha value determines how future oriented the AI is.
    #bigger number (up to 1) -> more future oriented
    def learnBatch(self, batch, alpha=0.9):
        batch = np.array(batch)
        # actions = batch[:, 2].reshape(self.batchSize).tolist()
        actions = batch[:, -3].astype(int).tolist()
        # rewards = batch[:, 3].reshape(self.batchSize).tolist()
        rewards = batch[:, -2].tolist()

        # stateToPredict = batch[:, 0].reshape(self.batchSize).tolist()
        stateToPredict = batch[:, 0 : np.product(self.tensorShape)].tolist()
        # nextStateToPredict = batch[:, 1].reshape(self.batchSize).tolist()
        nextStateToPredict = batch[:, np.product(self.tensorShape) : -3].tolist()

        statePrediction = self.model.predict(np.reshape(
            stateToPredict, (self.batchSize, *self.tensorShape)))
        nextStatePrediction = self.model.predict(np.reshape(
            nextStateToPredict, (self.batchSize, *self.tensorShape)))
        statePrediction = np.array(statePrediction)
        nextStatePrediction = np.array(nextStatePrediction)

        for i in range(self.batchSize):
            action = actions[i]
            reward = rewards[i]
            nextState = nextStatePrediction[i]
            qval = statePrediction[i, action]
            if reward < -5: 
                statePrediction[i, action] = reward
            else:
                #this is the q learning update rule
                statePrediction[i, action] += alpha * (reward + 0.95 * np.max(nextState) - qval)

        self.xTrain.append(np.reshape(
            stateToPredict, (self.batchSize, *self.tensorShape)))
        self.yTrain.append(statePrediction)
        history = self.model.fit(
            self.xTrain, self.yTrain, batch_size=5, epochs=2, verbose=0)
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)
        self.loss.append(loss)
        self.xTrain = []
        self.yTrain = []