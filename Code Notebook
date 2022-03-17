# %% [markdown]
#  # **Basic TensorFlow Deep Neural Network to predict hand written numbers**

# %% [markdown]
# ## 1. Libraries 

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# %% [markdown]
# ## 1.2. Dataset Import

# %%
mnist = tf.keras.datasets.mnist

# %% [markdown]
# ## 1.3. Split data into train and test

# %%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %% [markdown]
# ## 1.3.1. Data visulaiztion

# %%
plt.imshow(x_train[0])
plt.show

# %%
plt.imshow(x_train[3])
plt.show

# %% [markdown]
# ## 1.3.2. Data Normaliztion

# %%
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# %%
x_train

# %% [markdown]
# ## 2. Model Setup

# %% [markdown]
# ### Model consist of 1 flatten input layer and 2 hidden layers with 128 hidden neurons and 1 output layer with 10 neurons(0-9)
# #### Model Optimizer is **ADAM** 
# #### Metrics to monitor is **Accuracy**
# #### Loss is **Sparse categorical entropy**

# %%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# %% [markdown]
# ### 2.1. Metrics check on test data 
# #### **Accuracy** = 97%
# #### **Loss** = 0.090
# #### **Note**: Val acc and Val must not be way too larger or smaller than the train acc and loss if this happens that means the model is overfitting 

# %%
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_acc, val_loss)


# %% [markdown]
# ### 2.2. Saving the model 

# %%
model.save('Num_Reader.model')
new_model = tf.keras.models.load_model('Num_Reader.model')


# %%
predictions = new_model.predict([x_test])
print(predictions)

# %% [markdown]
# ### 3. Model Test 

# %%
print(np.argmax(predictions[0]))

# %%
plt.imshow(x_test[0])
plt.show()

# %%
