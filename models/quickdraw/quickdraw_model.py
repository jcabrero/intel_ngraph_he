import json
import tensorflow as tf

data_path = 'gs://amld-datasets/zoo_img'
batch_size = 100
labels = [label.strip() for label in 
          tf.io.gfile.GFile('{}/labels.txt'.format(data_path))]
counts = json.load(tf.io.gfile.GFile('{}/counts.json'.format(data_path)))
train_steps = counts['train'] // batch_size
eval_steps = counts['eval'] // batch_size

feature_spec = {
    'label': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'img_64': tf.io.FixedLenFeature(shape=[64, 64], dtype=tf.int64),
}

def parse_example(serialized_example):
  features = tf.io.parse_single_example(serialized_example, feature_spec)
  label = tf.one_hot(tf.squeeze(features['label']), len(labels))
  img_64 = tf.cast(features['img_64'], tf.float32) / 255.
  return img_64, label

ds_train = tf.data.TFRecordDataset(
    tf.io.gfile.glob('{}/train-*'.format(data_path))
    ).map(parse_example).batch(batch_size)
ds_eval  = tf.data.TFRecordDataset(
    tf.io.gfile.glob('{}/eval-*'.format(data_path))
    ).map(parse_example).batch(batch_size)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64,)),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(ds_train, steps_per_epoch=train_steps, epochs=1)
print('eval: ', model.evaluate(ds_eval, steps=eval_steps))

model.save('linear.h5')