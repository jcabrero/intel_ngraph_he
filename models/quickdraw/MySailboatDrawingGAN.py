#!/usr/bin/env python
# coding: utf-8

# # Privacy Preserving Generative Model
# This notebook aims to create a generative model which works in Python3

# In[1]:
#from IPython.get_ipython import get_ipython

#get_ipython().system('pip install imageio')
#get_ipython().system('pip install matplotlib')
#get_ipython().system('pip install tqdm')


# In[2]:


import time, os

# Numpy
import numpy as np

# Tensorflow and keras layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import activations
from tensorflow.keras import backend

# To generate GIFs


import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time


#from tqdm import tqdm

from IPython import display
#tf.reset_default_graph()


# In[3]:


def preprocess_real_samples(samples):
    samples = samples.astype('float32').reshape(-1, 28, 28, 1)
    samples = (samples - samples.min()) / (samples.max() - samples.min())
    return samples

def plot_4_by_4_images(x, save = False, savefile="img.png"):
    w=10
    h=10
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        img = np.random.randint(x.shape[0])
        fig.add_subplot(rows, columns, i)
        plt.imshow(x[i - 1, :, :, 0], cmap='gray')
    if save:
        plt.savefig(savefile)
    plt.show()
    
dataset = np.load('sailboat.npy')
dataset = preprocess_real_samples(dataset) 
plot_4_by_4_images(dataset)


# In[4]:


def generator_model():
    def square_activation(x):
        return x * x
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(Activation(square_activation))

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(Activation(square_activation))    

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(Activation(square_activation))
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model
noise = np.random.normal(0, 1, [1, 100])
generator = generator_model()
generator.summary()
generated_image = generator.predict(noise)
print(generated_image.shape)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')


# In[5]:


def discriminator_model():
    def square_activation(x):
        return x * x
    
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    #model.add(Activation(square_activation))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    #model.add(Activation(square_activation))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
discriminator = discriminator_model()
discriminator.summary()
pred= discriminator.predict(generated_image)
print(pred, pred > 0.5)


# In[6]:


def gan_model(generator, discriminator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = gan_model(generator, discriminator)
gan.summary()


# In[7]:


def generate_noise_samples(n=1):
    noise = np.random.normal(0, 1, (n, 100))
    return noise
def generate_fake_samples(n=1):
    noise = generate_noise_samples(n)
    X = generator.predict(noise)
    return X


# In[8]:


def generate_and_save_images(noise_test, epoch, batch):
    display.clear_output(wait=True)
    fake_images = generator.predict(noise_test)
    plot_4_by_4_images(fake_images, save=True, savefile='sailboat/i_{:04d}_{:04d}.png'.format(epoch, batch))


# In[9]:


def train_step(real_images, batch_size=64):
        real_label = np.ones((batch_size, 1))
        generated_images = generate_fake_samples(batch_size)
        generated_labels = np.zeros((batch_size, 1))
        X_dis = np.concatenate([real_images, generated_images])
        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size]=0.9
        
        discriminator.trainable = True
        discriminator.train_on_batch(X_dis, y_dis)
        #discriminator.train_on_batch(x_fake, y_fake)
        
        discriminator.trainable = False
        x_gan = generate_noise_samples(batch_size)
        y_gan = np.ones((batch_size, 1)) # We assume that we wanted true as answer from the discriminator
        gan.train_on_batch(x_gan, y_gan)


# In[ ]:


def train(dataset, epochs=50, batch_size=64):
    m = dataset.shape[0]
    m_batch = m // batch_size
    noise_test = np.random.normal(0,1, [20, 100])
    toc = time.time()
    for epoch in range(epochs):
        tic = time.time()
        for batch_num in range(m_batch):
            if batch_num % 10 == 0:
                #print ('[{}%] Time for epoch {} is {} sec'.format((batch_num / m_batch) * 100,epoch + 1, time.time()-tic), end='\r')
                #generate_and_save_images(noise_test, epoch, batch_num)
                print("[%0.2f%%] Time for epoch %d is %f sec" % ( (batch_num / m_batch) * 100, epoch + 1, time.time()-tic), end='\r')    
            batch_slot = batch_size * batch_num
            batch = dataset[batch_slot: batch_slot + batch_size]
            train_step(batch, batch_size)
        #Extra
        
        if time.time() - toc  > 30:
            toc = time.time()
            generate_and_save_images(noise_test, epoch, batch_num)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-tic))
    
    
train(dataset[:30], 10, 32)


# In[ ]:


generator.save('generator_sailboat_b.h5')
discriminator.save('discriminator_sailboat_b.h5')
gan.save('gan_sailboat_b.h5')


# ### EXTRA CODE

def make_gif(anim_file):
    #anim_file = 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob('sailboat/i_*.png')
      filenames = sorted(filenames)
      last = -1
      for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)

make_gif('apple.gif')