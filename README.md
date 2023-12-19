# FCCMachineLearning

# Supervised learning (classification/MAGIC)

## 1. DataSet:
using Magic dataset from https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

firstly we will import some dependency: 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

now upload dataset to google colab, and we can show it by defining the header with this code :
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
df.head()

then to change the class that have value g to integer(1), use this code: df["class"] = (df["class"] == "g").astype(int)

next we can show data as diagram for each criteria by the class type with code: for label in cols[:-1]:
  plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.5, density=True)
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.5, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  plt.show()

density make the data same, exam if 0 is 100 and 1 is 50, so will take 50


## 2. Train, validation, test dataset
create destructuing variable train, valid, and test with value np.split like: train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

then create new function name scale_dataset that get argument dataframe
get x by dataframe last column, and rest of it become y:   X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

then use scaler by StandartScaler to fit and transform x:   scaler = StandardScaler()
  X = scaler.fit_transform(X)

next create data as one 2d numpy array use hstack x and y, and we need to call np.reshape:   data = np.hstack((X, np.reshape(y, (-1, 1))))

then we can return the data, x, and y:   return data, X, y 

that is the function, next we will check sum of data with class 1 and 0 use: print(len(train[train["class"] == 1]))
print(len(train[train["class"] == 0]))

we will see the data is not equal. so we can oversample, to increase the number of less to make it equal, use imblearn.over_sampling from RandomOverSampler: from imblearn.over_sampling import RandomOverSampler

then add new parameter on scale_dataset function name oversample with value false as default: def scale_dataset(dataframe, oversample = False):

the add conditional if oversample, then use RandomOverSampler name ros, and fit_resample the x and y:   if oversample:
    ros = RandomOverSampler()
    x, y = ros.fit_resample(x, y)

create new code block, that will make variable with destructuring name train, x_train, y_train with value of scale_dataset function that passing train and oversample = true: train, X_train, y_train = scale_dataset(train, oversample=True)

and also do it with valid and test, but with oversampe=false: valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)


## 3. Model Classification
### 1. K Nearest Neighbor
to use it we will import package from sklearn: from sklearn.neighbors import KNeighborsClassifier

then use it as knn_model by passing parameter how many k will be use: knn_model  = KNeighborsClassifier(n_neighbors=5) 

then we can use it to training the data by use fit and passing x_train an y_train: knn_model.fit(X_train, y_train)

to do predict or use test data, we can make new variable name y_pred that value knn_model.predict that passing argument of x_text: y_pred = knn_model.predict(X_test)

to see the classification report, import it from sklearn: from sklearn.metrics import classification_report

and use it to see report of y_test, and the y_pred that use before: print(classification_report(y_test, y_pred))


### 2. naive bayes
we also import naive bayes from sklearn with code: from sklearn.naive_bayes import GaussianNB

and use it as nb_model: nb_model = GaussianNB()

the we can use train data with fit: nb_model = nb_model.fit(X_train, y_train)

then make prediction name y_pred using nb_model, also dont forget to do classification_report with y_test and y_pred: y_pred = nb_model.predict(X_test)

to show the report use code: print(classification_report(y_test, y_pred))


### 3. logistic regression
firstly we can import logistic regression model from sklearn: from sklearn.linear_model import LogisticRegression

and use it as variable lg_model and fit it with x_train and y_train: lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

then we can predict with that lg_model and see the report: y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))


### 4.svm:
same as before we can import svc(svm classification) model from sklearn: from sklearn.linear_model import LogisticRegression

and use it as vairable name svm_model and fit with our train data: lg_model = LogisticRegression()
lg_model.fit(X_train, y_train)

then we make y_pred that predict the x_test with svm_model and show the report: y_pred = lg_model.predict(X_test)
print(classification_report(y_test, y_pred))


### 5. neural network:
different from 4 model before, nn use some complicated method in it with tensorflow

we will use neural network classification with tensorflow, we can import tensorflow as tf: import tensorflow as tf

and make variable name nn_model with value tf.keras.Sequental and passing an array layer with node 32 and actiivation of relu also input_shape 10. and repeat it once without shape, then make last layer with node 1 and activation sigmoid:  nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
and we can compile it and passing optimizer adam from tf with 0.001, with loss binary_crossentropy, also metric an array with accuracy to see the accuration: nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

then take function from tensorflow fot plot_loss and plot_accuracy, place the code block after import: def plot_loss(history):
  plt.plot(history.history["loss"], label="loss")
  plt.plot(history.history["val_loss"], label = "val_loss")
  plt.xlabel("Epoch")
  plt.ylabel("Binary crossentropy")
  plt.legend()
  plt.grid(True)
  plt.show()

def plot_accuracy(history):
  plt.plot(history.history["accuracy"], label="accuracy")
  plt.plot(history.history["val_accuracy"], label = "val_accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Binary crossentropy")
  plt.legend()
  plt.grid(True)
  plt.show()


next create a variable nam history with value of nn_model that fit the x_train, y_train, epoch of 100, batch_size of 32, validation_split of 0.2 because tensorflow use validation it self, and verbose of 0: history = nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

so we can wait train process, if done we can use plot_loss and plot_accuracy by passing history: plot_loss(history)
plot_accuracy(history)


to more optimize the model we can change the node in layers, next we will make the node change automaticly also change the optimizer, epoch, batch_size, etc.

add new Dropout layers for after each layer expect last layer to choose node and dont train it to prevent overfit:    nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(32, activation="relu", input_shape=(10,)),
      tf.keras.layers.Dropout(),
      tf.keras.layers.Dense(32, activation="relu"),
      tf.keras.layers.Dropout(),
      tf.keras.layers.Dense(1, activation="sigmoid")
  ])

then wrap the nn_model on function name train_model with some argument(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, and epochs: def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):

change the node with num_nodes:       tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),

passing the dropout_prob to Dropout:       tf.keras.layers.Dropout(dropout_prob),

change optimizers value with lr:   nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])

place history variable on this function, and set some value on fit like epochs equal epochs, batch_size equal batch_size:   history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

so return the nn_model and hitsory:   return nn_model, history

this is the full function: def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes, activation="relu"),
      tf.keras.layers.Dropout(),
      tf.keras.layers.Dense(1, activation="sigmoid")
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])

  history = nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

  return nn_model, history



try change some value with looping

use epochs of 100: epochs=100

and create loop with for, with 16, 32, and 64 for num_nodes: for num_nodes in [16, 32, 64]:

0, 0.2 for dropout_prob:   for dropout_prob in[0, 0.2]:

lf with value 0.01, 0.005. 0.002:     for lr in [0.01, 0.005, 0.001]:

for batch_size with 32, 64, 128:       for batch_size in [32, 64, 128]:

then on that nesting loop destructuring the variable model and history from train_model function by passing all the argument:         model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)

next for plot_loss and plot_accuracy
we can combine the plot_loss and plot_accuacy
back to plot_loss change the name to plot_history: def plot_history(history):

split the plot with ax1 and ax2: 

use ax1 as plot loss, and ax2 as plot accuracy, also change the x and ylabel to set_x and set_ylabel: def plot_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)

  plt.show()

so we can use this function to plot it:         plot_history(history)


next to the loop, we can see what is out parameter by print it:         print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")

and create variable name val_loss with value model.evaluate of x_valid and y_valid: 


on top of iteration, crate least_val_loss with value float inf: least_val_loss = float('inf')

and least_loss_model of none: least_loss_model = None

back to the iteration after val_loss add conditioal that check if val_loss less than least_val_loss, we will set least_val_loss with val_loss and also for least_loss_model with model:         if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model

this the full code of loop: least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [16, 32, 64]:
  for dropout_prob in[0, 0.2]:
    for lr in [0.01, 0.005, 0.001]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history)
        val_loss = model.evaluate(X_valid, y_valid)[0]
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model

we can change the validation_split to validation_data with value valid, for now we will still use validation_split: validation_data=valid


lastly we can predict with least_loss_model by passing x_test: y_pred = least_loss_model.predict(X_test)

and we can caxt them with conditional list, if y_pred greater than 0.5 change it as int, and reshape it to one dimensioal with -1: y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

next we can check the report by passing y_test and y_pred: print(classification_report(y_test, y_pred))




## 4. Model Regression
