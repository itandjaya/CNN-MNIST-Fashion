## NN Fashin Classification.
## Data from MNIST Fasion: 60k training sets, and 10k test sets.
## https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md#get-the-data

## Classification Labels:
## Label	Description
##  0	    T-shirt/top
##  1	    Trouser
##  2	    Pullover
##  3	    Dress
##  4	    Coat
##  5	    Sandal
##  6	    Shirt
##  7	    Sneaker
##  8	    Bag
##  9	    Ankle boot

from __future__ import division, absolute_import, print_function, unicode_literals;

import os;
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;    #   To display image - gray scale.


from tensorflow.keras.layers import Dense, Flatten, Conv2D;
from tensorflow.keras.regularizers import l1 as l1_reg, l2 as l2_reg;
from tensorflow import keras;
from random import randint;

## Import datasets locally
from import_data import import_urls;
#from tensorflow.examples.tutorials.mnist import input_data;


"""
class CNN_model(    Model):

    def __init__(   self):

        super(  CNN_model, self).__init__();
        self.conv1 = Conv2D(32, 3, activation='relu');
        self.flatten1 = Flatten();
        self.d2 = Dense(128, activation='relu');
        self.d3 = Dense(10);

    def call(   self, x):
        x = self.conv1(x);
        x = self.flatten1(x);
        x = self.d2(x);
        h = self.d3(x);

        return h;
"""

fashion_labels = {  0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat",
                    5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"  }; 

BATCH_SIZE = 128;

loss_fn     = tf.keras.losses.SparseCategoricalCrossentropy(    from_logits=True);
opt_Adam    = tf.keras.optimizers.Adam();

train_loss  = tf.keras.metrics.Mean(name='train_loss');
test_loss   = tf.keras.metrics.Mean(name='test_loss');

train_accuracy  = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy');
test_accuracy   = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy');

EPOCHS  =   5;


@tf.function
def train_step( model, batch_X_train, batch_y_train):

    with tf.GradientTape() as tape:

        predictions =   model(  batch_X_train, training = True  );
        loss        =   loss_fn(    batch_y_train, predictions);
    
    gradients   =   tape.gradient(  loss,   model.trainable_variables);

    opt_Adam.apply_gradients(      zip(    gradients, model.trainable_variables));

    train_loss(loss);
    train_accuracy(batch_y_train, predictions);

    return;

@tf.function
def test_step(  model,  batch_X_test, batch_y_test):

    predictions =   model(  batch_X_test, training = False);
    loss        =   loss_fn(    batch_y_test, predictions);

    test_loss(loss);
    test_accuracy(  batch_y_test, predictions);

    return;




def plot_image(x, class_label):
    ##  Using matplotlib to display the gray-scaled digit image.
    ##  Input:  2D np.darray representing (28x28 pixels).

    image = np.asarray(x).squeeze();
    plt.text(   0, 10,    fashion_labels[class_label], bbox=dict(facecolor='white', alpha=1));

    plt.imshow(image);
    plt.show();
    return;


def main():

    
    ##  Set redownload_data = True for initial download.
    (X_train, y_train), (X_test, y_test) = import_urls( redownload_data = False);  

    ## Checking gray-scaled images to if data upload is done correctly.

    #for _ in range(10):
    #    i   =   randint(0, 60000);
    #    img, label  =   X_train[i], y_train[i][0];
    #    plot_image( img, label); 
    #return;

    ## Normalize data to avoid overflow/over-rounding numbers.
    X_avg, X_range  =   np.mean(X_train), ( np.max(X_train) - np.min(X_train) );
    X_train         =   (   (X_train - X_avg)/X_range   ).astype(np.float32);
    X_test          =   (   (X_test  - X_avg)/X_range   ).astype(np.float32);
    

    ## Input data size: 60k x 28 x 28. Output data: 60k x 1.
    #print("Training data size: ", np.shape(x_train), np.shape(y_train));

    # Add a channels dimension - due to using Conv2D.
    X_train = X_train[..., tf.newaxis];
    X_test  = X_test[..., tf.newaxis];

    #print("\n", np.shape(X_train), np.shape(y_train));
    #print(np.shape(X_test), np.shape(y_test));

    ## Batch the training and test dataset.
    train_ds = tf.data.Dataset.from_tensor_slices(
                (X_train, y_train)).shuffle(10000).batch(32);

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32);


    ## Build the CNN model:

    #model   =   CNN_model();
    
    model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(     32, 3, activation = 'relu'),##  Convolution layer.

            tf.keras.layers.Flatten(), 

            tf.keras.layers.Dense(      196, activation='relu',     
                                        kernel_regularizer=l2_reg(l=0.03)),                                            
            tf.keras.layers.Dropout(    0.2),                      ## Dropouts to prevent overfitting.

            tf.keras.layers.Dense(      49, activation='relu',     
                                        kernel_regularizer=l2_reg(l=0.03)),                                            
            tf.keras.layers.Dropout(    0.2), 

            tf.keras.layers.Dense(      10),                        ## Output layer (10-digit classifications).
            ]);

    model.compile(  #optimizer   =   'adam',
                    #loss        =   loss_fn,
                    metrics     =   ['accuracy']);    


    for epoch in range( EPOCHS):
        
        #   Resets all of the metric state variables per epoch.
        train_loss.reset_states();        
        test_loss.reset_states();

        train_accuracy.reset_states();
        test_accuracy.reset_states();

        for x_train, y_train in train_ds:
            train_step( model, x_train, y_train);

        for x_test, y_test in test_ds:
            test_step(  model, x_test, y_test);

        str_tmpl =      '\nEpoch {0}, Loss: {1:.5f}, Accuracy: {2:.2f} %, ' + \
                        'Test Loss: {3:.5f}, Test Accuracy: {4:.2f} %';
        print(          str_tmpl.format(    epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100,
                        ));

    model.summary();
    return 0;


if __name__ == '__main__':      main();