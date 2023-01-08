#!/usr/bin/env python
# coding: utf-8

# In[2]:



import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


# # Specify each path

# In[ ]:


dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'


# # Set number of classes

# In[ ]:


NUM_CLASSES = 10


# # Dataset reading

# In[ ]:


X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))


# In[ ]:


y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)


# # Model building

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])


# In[ ]:


model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


# Model checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# Callback for early stopping
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)


# In[ ]:


# Model compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# # Model training

# In[ ]:


model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)


# In[ ]:


# Model evaluation
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)


# In[ ]:


# Loading the saved model
model = tf.keras.models.load_model(model_save_path)


# In[ ]:


# Inference test
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


# # Confusion matrix

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)


# # Convert to model for Tensorflow-Lite

# In[ ]:


# Save as a model dedicated to inference
model.save(model_save_path, include_optimizer=False)


# In[ ]:


# Transform model (quantization)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)


# # Inference test

# In[ ]:


interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()


# In[ ]:


# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[ ]:


interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Inference implementation\ninterpreter.invoke()\n tflite_results = interpreter.get_tensor(output_details[0]['index'])")


# In[ ]:


print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

