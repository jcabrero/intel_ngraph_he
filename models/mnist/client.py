#!/usr/bin/env python
# coding: utf-8

import time
import argparse
import numpy as np
import sys
import os

import pyhe_client

batch_size = 1024
(x_train, y_train, x_test, y_test) = load_mnist()
data = x_test[:batch_size].flatten("C")
print("A")
client = pyhe_client.HESealClient('localhost', 34000, batch_size,
                                  {'import/input': ("encrypt", data)})

print("B")
results = np.round(client.get_results(), 2)
y_pred_reshape = np.array(results).reshape(batch_size, 10)
with np.printoptions(precision=3, suppress=True):
    print(y_pred_reshape)

y_pred = y_pred_reshape.argmax(axis=1)
print("y_pred", y_pred)

correct = np.sum(np.equal(y_pred, y_test.argmax(axis=1)))
acc = correct / float(FLAGS.batch_size)
print("correct", correct)
print("Accuracy (batch size", FLAGS.batch_size, ") =", acc * 100.0, "%")

