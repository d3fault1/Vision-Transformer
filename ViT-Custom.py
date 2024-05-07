import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as mp
import scipy.interpolate as ip
import numpy as np
import os
import sys
import timeit
import seaborn as sns

matplotlib.use('Agg')
mp.ioff()

def update_progress(progress, loss, acc):
    barLength = 40  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1}% - loss: {3} - acc: {4} {2}".format("=" * (block - 1) + ">" + "-" * (barLength - block), round(progress * 100, 2), status, round(loss, 3), round(acc, 3))
    sys.stdout.write(text)
    sys.stdout.flush()
    return

class Vit_Custom_ExtraLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(Vit_Custom_ExtraLayer, self).__init__()
        return
    def build(self, input_shape):
        outer_dim = input_shape[-1]
        self.w = self.add_weight(name='extra_w', shape=(1, 1, outer_dim), initializer=tf.random_normal_initializer(), trainable=True)
        return
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        broadcasted = tf.broadcast_to(self.w, (batch_size, 1, inputs.shape[-1]))
        casted_broadcast = tf.cast(broadcasted, inputs.dtype)
        return tf.concat([inputs, casted_broadcast], axis=1) #shape (batch_size, no_patches + 1, new_size)
    def get_learnable_patch(self):
        return self.w

class ViT_Custom_PatchEncode(keras.layers.Layer):
    def __init__(self, patch_size, output_dim=None, padding='VALID', shrink_ratio=1):
        super(ViT_Custom_PatchEncode, self).__init__()
        self.patch_size = patch_size
        self.padding = padding
        self.shrink_ratio = shrink_ratio
        self.output_dim = output_dim
        return

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.no_patches = int((input_shape[1] / self.patch_size) ** 2)
        self.flattend_size = int(self.patch_size * self.patch_size * self.num_channels)
        if (self.output_dim == None): self.new_size = int(self.flattend_size / self.shrink_ratio)
        else: self.new_size = self.output_dim
        self.linear_w1 = self.add_weight(name='w', shape=(self.flattend_size, self.new_size),
                                 initializer=tf.initializers.random_uniform(), trainable=True)
        self.linear_b1 = self.add_weight(name='b', shape=(self.new_size,),
                                 initializer=tf.initializers.random_uniform(), trainable=True)
        self.embeddings = self.add_weight(name='embeddings', shape=(self.no_patches, self.new_size),
                                 initializer=tf.initializers.random_uniform(), trainable=True)
        return
    
    def call(self, input, Training=True):
        # input shape (batch_size, im_size, im_size, num_channels)
        patches = tf.image.extract_patches(input, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1],
                                            rates=[1, 1, 1, 1], padding=self.padding) # shape (batch_size, new_im_size, new_im_size, patch_size * patch_size * num_channels)
        patches_flattened = tf.reshape(patches, shape=(-1, self.no_patches, self.flattend_size)) # shape (batch_size, no_patches, flattened_size)
        patch_linearized = tf.matmul(patches_flattened, self.linear_w1) + self.linear_b1 # shape (batch_size, no_patches, new_size)
        positions = tf.range(0, limit=self.no_patches, delta=1)
        positions_embedding = tf.nn.embedding_lookup(self.embeddings, positions) # shape (no_patches, new_size)
        patch_encoded = patch_linearized + positions_embedding # shape (batch_size, no_patches, new_size)
        return patch_encoded
    
class ViT_Custom_MultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_heads, key_dim=None, shrink_ratio = 4):
        super(ViT_Custom_MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.shrink_ratio = shrink_ratio
        self.key_dim = key_dim
        return

    def build(self, input_shape):
        # shape (batch_size, no_patches, flattened_size)
        self.no_patches = input_shape[1]
        self.flattened_size = input_shape[-1]
        self.new_size = int(self.flattened_size / self.shrink_ratio)
        if (self.key_dim == None): self.key_dim = input_shape[-1]
        self.query_w = self.add_weight(name='query_w', shape=(self.flattened_size, self.new_size, self.num_heads),
                                        initializer=tf.initializers.random_uniform(), trainable=True)
        self.key_w = self.add_weight(name='key_w', shape=(self.flattened_size, self.new_size, self.num_heads),
                                        initializer=tf.initializers.random_uniform(), trainable=True)
        self.value_w = self.add_weight(name='value_w', shape=(self.flattened_size, self.new_size, self.num_heads),
                                        initializer=tf.initializers.random_uniform(), trainable=True)
        self.out_w1 = self.add_weight(name='out_w1', shape=(int((self.num_heads * self.no_patches)), self.no_patches),
                                        initializer=tf.initializers.random_uniform(), trainable=True)
        self.out_w2 = self.add_weight(name='out_w2', shape=(self.new_size, self.flattened_size),
                                        initializer=tf.initializers.random_uniform(), trainable=True)
        self.query_b = self.add_weight(name='query_b', shape=(self.new_size, self.num_heads),
                                        initializer=tf.initializers.zeros(), trainable=True)
        self.key_b = self.add_weight(name='key_b', shape=(self.new_size, self.num_heads),
                                        initializer=tf.initializers.zeros(), trainable=True)
        self.value_b = self.add_weight(name='value_b', shape=(self.new_size, self.num_heads),
                                        initializer=tf.initializers.zeros(), trainable=True)
        self.out_b1 = self.add_weight(name='out_b1', shape=(self.new_size, self.no_patches),
                                        initializer=tf.initializers.zeros(), trainable=True)
        self.out_b2 = self.add_weight(name='out_b2', shape=(self.no_patches, self.flattened_size),
                                        initializer=tf.initializers.zeros(), trainable=True)
        return
    
    def call(self, query, value, Training=True):
        key = tf.identity(query)
        query_linearized = tf.tensordot(query, self.query_w, axes=[[2], [0]]) + self.query_b # shape (batch_size, no_patches, new_size, num_heads)
        key_linearized = tf.tensordot(key, self.key_w, axes=[[2], [0]]) + self.key_b # shape (batch_size, no_patches, new_size, num_heads)
        value_linearized = tf.tensordot(value, self.value_w, axes=[[2], [0]]) + self.value_b # shape (batch_size, no_patches, new_size, num_heads)
        attn_matrix = tf.matmul(tf.transpose(query_linearized, perm=[0, 3, 1, 2]), tf.transpose(key_linearized, perm=[0, 3, 2, 1])) # shape (batch_size, num_heads, no_patches, no_patches)
        attn_matrix_scaled = attn_matrix / np.sqrt(self.key_dim)
        attn_matrix_final = tf.nn.softmax(attn_matrix_scaled)
        self.attn = attn_matrix_scaled
        output = tf.transpose(tf.matmul(attn_matrix_final, tf.transpose(value_linearized, perm=[0, 3, 1, 2])), perm=[0, 3, 2, 1]) # shape (batch_size, new_size, no_patches, num_heads)
        #output_concatenated = tf.reshape(output, shape=(-1, self.new_size, self.no_patches * self.num_heads)) # shape (batch_size, new_size, no_patches * num_heads)
        output_concatenated = tf.concat(tf.unstack(output, axis=-1), axis=-1) # shape (batch_size, new_size, no_patches * num_heads)
        output_final_intermediate = tf.matmul(output_concatenated, self.out_w1) + self.out_b1 # shape (batch_size, new_size, no_patches)
        output_final_transposed = tf.transpose(output_final_intermediate, perm=[0, 2, 1]) # shape (batch_size, no_patches, new_size)
        output_final = tf.matmul(output_final_transposed, self.out_w2) + self.out_b2 # shape (batch_size, no_patches, flattened_size)
        return output_final
    
    def get_attention(self):
        return self.attn

class ViT_Custom_BatchNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(ViT_Custom_BatchNormalization, self).__init__()
        self.epsilon = epsilon
        return
    
    def call(self, input, Training=True):
        mean, variance = tf.nn.moments(input, [0, 1, 2])
        output = tf.nn.batch_normalization(input, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=self.epsilon)
        return output

class ViT_Custom_LayerNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(ViT_Custom_LayerNormalization, self).__init__()
        self.epsilon = epsilon
        return
    
    # def build(self, input_shape):
    #     outer_dim = input_shape[-1]
    #     self.w = self.add_weight(name='norm_w', shape=(outer_dim, outer_dim), initializer=tf.ones_initializer(), trainable=True)
    #     self.b = self.add_weight(name='norm_b', shape=outer_dim, initializer=tf.zeros_initializer(), trainable=True)
    #     return

    def call(self, input):
        mean, variance = tf.nn.moments(input, [0, 1, 2])
        normalized = (input - mean) / tf.sqrt(variance + self.epsilon)
        # if (input.shape.rank > 3): output = tf.tensordot(normalized, self.w, axes=[[self.rank - 1], [0]]) + self.b
        # else: output = tf.matmul(normalized, self.w) + self.b
        return normalized

class ViT_Custom_Dropout(keras.layers.Layer):
    def __init__(self, rate=0.5):
        super(ViT_Custom_Dropout, self).__init__()
        self.rate = rate
        return
    
    def call(self, input, Training=True):
        if (Training):
            output = tf.nn.dropout(input, rate=self.rate)
        else:
            output = input
        return output
    
class ViT_Custom_Linear(keras.layers.Layer):
    def __init__(self, units=None, activation=None, shrink_ratio = 1.0):
        super(ViT_Custom_Linear, self).__init__()
        self.shrink_ratio = shrink_ratio
        self.units = units
        self.activation = activation
        return
    
    def build(self, input_shape):
        self.rank = input_shape.rank
        self.last_dim = input_shape[-1]
        if (self.units != None): self.out_dim = int(self.units)
        else: self.out_dim = int(self.last_dim / self.shrink_ratio)
        self.w = self.add_weight(name='w', shape=(self.last_dim, self.out_dim), 
                                initializer=tf.initializers.random_uniform(), trainable=True)
        self.b = self.add_weight(name='b', shape=(self.out_dim,),
                                initializer=tf.initializers.random_uniform(), trainable=True)
        return
    
    def call(self, input, Training=True):
        if (self.rank > 3): output = tf.tensordot(input, self.w, axes=[[self.rank - 1], [0]]) + self.b
        else: output = tf.matmul(input, self.w) + self.b
        if (self.activation != None):
            self.activation = keras.activations.get(self.activation)
            output_final = self.activation(output)
        else: output_final = output
        return output_final

class ViT_Custom_Add(keras.layers.Layer):
    def __init__(self):
        super(ViT_Custom_Add, self).__init__()
        return
    
    def call(self, inputs, Training=True):
        return tf.add_n(inputs)

class ViT_Custom_Flatten(keras.layers.Layer):
    def __init__(self):
        super(ViT_Custom_Flatten, self).__init__()
        return

    def call(self, input, Training=True):
        flat_shape = tf.reduce_prod(input.shape[1:])
        return tf.reshape(input, shape=(-1, flat_shape))

class ViT_Custom_Model(keras.Model):
    def __init__(self, patch_size, num_classes, projection_dim, transformer_layers, num_heads, patch_padding, dense_units, transformer_dropout, feature_dropout, norm_epsilon):
        super(ViT_Custom_Model, self).__init__()
        self.transformer_layers = transformer_layers
        self.mha = []
        self.dense = []
        self.extra_layer = Vit_Custom_ExtraLayer()
        self.patch_encode = ViT_Custom_PatchEncode(patch_size=patch_size, output_dim=projection_dim, padding=patch_padding)
        #self.norm = ViT_Custom_BatchNormalization(epsilon=norm_epsilon)
        self.norm = ViT_Custom_LayerNormalization(epsilon=norm_epsilon)
        self.add = ViT_Custom_Add()
        self.flat = ViT_Custom_Flatten()
        self.transformer_drop = ViT_Custom_Dropout(rate=transformer_dropout)
        self.feature_drop = ViT_Custom_Dropout(rate=feature_dropout)
        for _ in range(self.transformer_layers):
            temp_list = []
            self.mha.append(ViT_Custom_MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim))
            temp_list.append(ViT_Custom_Linear(units=dense_units[0], activation='gelu'))
            temp_list.append(ViT_Custom_Linear(units=dense_units[1], activation='gelu'))
            self.dense.append(temp_list)
        temp_list = []
        temp_list.append(ViT_Custom_Linear(units=dense_units[2], activation='gelu'))
        temp_list.append(ViT_Custom_Linear(units=dense_units[3], activation='gelu'))
        temp_list.append(ViT_Custom_Linear(units=num_classes, activation='softmax'))
        self.dense.append(temp_list)
        return

    def call(self, input):
        output = self.patch_encode(input)
        output = self.extra_layer(output)
        for i in range(self.transformer_layers):
            skip = output
            output = self.norm(output)
            output = self.mha[i](output, output)
            output = self.add([output, skip])

            skip = output
            output = self.norm(output)
            output = self.dense[i][0](output)
            output = self.transformer_drop(output)
            output = self.dense[i][1](output)
            output = self.transformer_drop(output)
            output = self.add([output, skip])
        output = self.norm(output)
        output = self.flat(output)
        output = self.feature_drop(output)
        output = self.dense[self.transformer_layers][0](output)
        output = self.feature_drop(output)
        output = self.dense[self.transformer_layers][1](output)
        output = self.feature_drop(output)
        output = self.dense[self.transformer_layers][2](output)
        return output
    
    def get_attention_matrix(self):
        return [x.get_attention() for x in self.mha]
    
    def get_learnable_patch(self):
        return self.extra_layer.get_learnable_patch()

@tf.function
def train_on_batch(x, y, vit_model):
    with tf.GradientTape() as tape:
        y_true = tf.squeeze(y)
        y_pred = vit_model(x, training=True)
        vit_model.trainable = True
        tape.watch(vit_model.trainable_variables)
        loss = vit_model.loss(y_true, y_pred)
        loss = tf.math.reduce_mean(loss)
        gradients = tape.gradient(loss, vit_model.trainable_weights)
    vit_model.optimizer.apply_gradients(zip(gradients, vit_model.trainable_weights))
    return loss, y_pred, gradients

@tf.function
def validation_on_batch(x, y, vit_model):                     
    y_pred = vit_model(x, training=False)
    vit_model.trainable = False              
    loss = vit_model.loss(y, y_pred)                                           
    return loss, y_pred

@tf.function
def test_on_batch(x, vit_model):
    vit_model.trainable = False                    
    y_pred = vit_model(x, training=False)            
    return y_pred

def plot_accs(plot_iters, test_acc):
    mp.cla()
    mp.ylim(0.0, 1.0)
    mp.xlabel('No. of Iterations')
    mp.ylabel('Accuracy')
    plot_iters_smooth = np.linspace(np.min(plot_iters), np.max(plot_iters), 250)
    xy_spline = ip.make_interp_spline(plot_iters, test_acc)
    test_acc_smooth = xy_spline(plot_iters_smooth)
    mp.plot(plot_iters_smooth, test_acc_smooth)
    fig = mp.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig('./myrun/vit/mnist/vit_accuracy')
    return

def parse_dataset(config, dataset):
    if (dataset == 'cifar10'):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train, x_test = (x_train).astype(np.float32) / 255.0, (x_test).astype(np.float32) / 255.0
        tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train.astype(np.float32))).batch(config['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test.astype(np.float32))).batch(config['batch_size'])
        return tr_dataset, val_dataset
    if (dataset == 'cifar100'):
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        x_train, x_test = (x_train).astype(np.float32) / 255.0, (x_test).astype(np.float32) / 255.0
        tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train.astype(np.float32))).batch(config['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test.astype(np.float32))).batch(config['batch_size'])
        return tr_dataset, val_dataset
    if (dataset == 'mnist'):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train, x_test = (x_train).astype(np.float32) / 255.0, (x_test).astype(np.float32) / 255.0
        x_train, x_test = tf.expand_dims(x_train, axis=-1), tf.expand_dims(x_test, axis=-1)
        tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(config['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config['batch_size'])
        return tr_dataset, val_dataset

def model_compile(config):
    model = ViT_Custom_Model(patch_size=config['patch_size'], num_heads=config['num_heads'], transformer_layers=config['transformer_layers'],
                             num_classes=config['num_class'], projection_dim=config['projection_dim'], dense_units=config['dense_units'],
                             transformer_dropout=config['transformer_dropout'], feature_dropout=config['feature_dropout'], norm_epsilon=config['norm_epsilon'],
                             patch_padding=config['patch_padding'])

    # checkpoint_filepath = "/myrun/ViT_Custom/"
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     checkpoint_filepath,
    #     monitor="val_accuracy",
    #     save_best_only=True,
    #     save_weights_only=True
    # )

    model.compile(
        loss=tf.losses.sparse_categorical_crossentropy,
        #optimizer=tf.optimizers.SGD(learning_rate=0.001),
        # optimizer=tf.optimizers.Adam(learning_rate=0.005, weight_decay=0.0001),
        optimizer = tfa.optimizers.AdamW(learning_rate=0.005, weight_decay=0.0001),
        metrics=['acc']
        )
    return model

def model_fit(train_dataset, validation_dataset, vit_model, epochs, plot_freqs=15, save_path=''):
    total_step_num = train_dataset.cardinality().numpy()
    total_val_step_num = validation_dataset.cardinality().numpy()
    train_acc = []
    valid_acc = []
    train_err = []
    valid_err = []
    test_acc = []
    plot_iters = []
    start = timeit.default_timer()

    for epoch in range(epochs): 
        print('Epoch: ', epoch + 1, '/', epochs)

        # -------------Training--------------------
        curr_train_acc = []
        curr_train_loss = []
        for step, (x, y) in enumerate(train_dataset):
            loss, y_pred, gradients = train_on_batch(x, y, vit_model)
            loss = tf.reduce_mean(tf.cast(loss, tf.float32))
            curr_train_loss.append(loss.numpy())
            corr = tf.equal(tf.math.argmax(y_pred, axis=-1), tf.cast(tf.squeeze(y), dtype=tf.int64))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            curr_train_acc.append(accuracy.numpy())
            if ((epoch * total_step_num) + (step + 1)) % plot_freqs == 0:
                (test_x, test_y) = next(iter(validation_dataset))
                test_y_pred = test_on_batch(test_x, vit_model)
                test_corr = tf.equal(tf.math.argmax(test_y_pred, axis=-1), tf.cast(tf.squeeze(test_y), dtype=tf.int64))
                acc = tf.reduce_mean(tf.cast(test_corr, tf.float32))
                test_acc.append(acc.numpy())
                plot_iters.append((epoch * total_step_num) + (step + 1))
            update_progress((step + 1) / int(total_step_num), float(np.mean(curr_train_loss)), float(np.mean(curr_train_acc)))
        train_acc.append(np.mean(curr_train_acc))
        train_err.append(np.mean(curr_train_loss))
        print()
        print('Training Acc  ', train_acc[epoch])
        print('Training loss  ', train_err[epoch])

        # ---------------Validation----------------------
        curr_val_acc = []
        curr_val_loss = []
        for step, (x, y) in enumerate(validation_dataset):           
            vloss, y_pred  = validation_on_batch(x, y, vit_model)
            vloss = tf.reduce_mean(tf.cast(vloss, tf.float32))
            curr_val_loss.append(vloss.numpy())
            corr = tf.equal(tf.math.argmax(y_pred, axis=-1), tf.cast(tf.squeeze(y), dtype=tf.int64))
            va_accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            curr_val_acc.append(va_accuracy.numpy())
            update_progress((step + 1) / int(total_val_step_num), float(np.mean(curr_val_loss)), float(np.mean(curr_val_acc)))
        valid_acc.append(np.mean(curr_val_acc))
        valid_err.append(np.mean(curr_val_loss))
        print()
        print('Validation Acc  ', valid_acc[epoch])
        print('Validation loss  ', valid_err[epoch])
        vit_model.save_weights(save_path + 'vit_custom_model.h5')

    stop = timeit.default_timer()
    vit_model.save_weights(save_path + 'vit_custom_model.h5')
    
    plot_accs(plot_iters, test_acc)
    print()
    print('Total Training Time: ', stop - start)
    print('Training Acc   ', train_acc[epoch])
    print('Validation Acc ', valid_acc[epoch])            
    print('------------------------------------')
    print('Training error   ', train_err[epoch])
    print('Validation error',  valid_err[epoch])
    return vit_model

def model_test(test_dataset, vit_model):
    vit_model.trainable = False
    total_step_num = test_dataset.cardinality().numpy()  
    true_y = []
    y_preds = []
    acc_test = []

    for step, (x, y) in enumerate(test_dataset):
        true_y.append(y)
        y_pred = test_on_batch(x, vit_model)
        y_preds.append(y_pred)
        corr = tf.equal(tf.math.argmax(y_pred, axis=-1), tf.cast(tf.squeeze(y), dtype=tf.int64))
        accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
        acc_test.append(accuracy.numpy())
        update_progress((step + 1) / int(total_step_num), 0, np.mean(acc_test))
    test_acc = np.mean(acc_test)

    print()
    print('Test accuracy : ', test_acc)
    print('Best Test accuracy : ', np.amax(acc_test))
    return

def model_predict(dataset, vit_model):
    total_step = dataset.cardinality().numpy()
    y_preds = []
    for step, (x, y) in enumerate(dataset):
        y_pred = test_on_batch(x, vit_model)
        y_preds.append(y_pred)
    return np.array(y_preds)


def main_function():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #tf.autograph.set_verbosity(2)

    config = {}
    config['batch_size'] = 64
    config['patch_size'] = 7
    config['num_heads'] = 12
    config['transformer_layers'] = 8
    config['projection_dim'] = 128
    config['dense_units'] = [256, 128, 2048, 1024]
    config['num_class'] = 10
    config['transformer_dropout'] = 0.1
    config['feature_dropout'] = 0.5
    config['norm_epsilon'] = 1e-6
    config['patch_padding'] = 'VALID'

    tr_dataset, val_dataset = parse_dataset(config=config, dataset='mnist')
    model = model_compile(config=config)

    model.build(input_shape=(None, 28, 28, 1))

    # model.load_weights('./myrun/vit/' + 'vit_custom_model').expect_partial()
    model_fit(train_dataset=tr_dataset, validation_dataset=val_dataset, vit_model=model, epochs=10, plot_freqs=100, save_path='./myrun/vit/mnist/')
    # model.save_weights('vit_custom_mnist_smaller.h5')
    
    # model.build(input_shape=(None, 32, 32, 3))
    # model.load_weights('vit_custom_mnist.h5')

    # model_test(val_dataset, model)
    # model_predict(val_dataset, model)

    # Print Attention Matrix
    exit()
    
    val_enum = iter(val_dataset)
    for i in range(5):
        (x, y) = next(val_enum)
        model.trainable = False
        y_pred = model(x, training=False)
        
        # tf.print(tf.argmax(y_pred, axis=1))

        mp.imshow(x[0])
        mp.gcf().set_size_inches(10, 10)
        mp.gcf().savefig('attention_matrices3\\batch_{}_input_image.png'.format(i))
        mhas = model.get_attention_matrix()
    #     # mhas_tensor = tf.stack(mhas, axis=1)
    #     # head_reduced = tf.reduce_mean(mhas_tensor, axis=2)
    #     # layer_summed = tf.reduce_sum(head_reduced, axis=1)
    #     # mp.imshow(layer_summed[0])
    #     # mp.gcf().set_size_inches(10, 10)
    #     # mp.gcf().savefig('attention_matrices2\\batch_{}_attention_matrix.png'.format(i))
        
        i_list = [tf.eye(mhas[0].shape[-1], dtype=tf.float32) for _ in range(mhas[0].shape[0])]
        rollout = tf.stack(i_list, axis=0)

        for mha in mhas:
            mha_reduced = tf.reduce_mean(mha, axis=1)
            rollout = tf.matmul(mha_reduced, rollout)
    #     # for attn_layer_no in range(len(mhas)):
    #     #     plottable_attn = tf.reduce_mean(mhas[attn_layer_no], axis=1)
    #     #     mp.imshow(plottable_attn[0])
    #     #     mp.gcf().set_size_inches(10, 10)
    #     #     mp.gcf().savefig('attention_matrices2\\batch_{}_attention_matrix_layer_{}.png'.format(i, attn_layer_no))

    #         # for head_no in range(len(mhas[attn_layer_no][0])):
    #         #     mp.imshow(mhas[attn_layer_no][0][head_no])
    #         #     mp.gcf().set_size_inches(10, 10)
    #         #     mp.gcf().savefig('attention_matrices2\\batch_{}_attention_matrix_layer_{}_head_{}.png'.format(i, attn_layer_no, head_no))
    #             # mp.imsave('attention_matrices2\\attention_matrix_layer_{}_head_{}.png'.format(attn_layer_no, head_no), mhas[attn_layer_no][0][head_no])
            
        mp.imshow(rollout[0])
        mp.gcf().set_size_inches(10, 10)
        mp.gcf().savefig('attention_matrices3\\batch_{}_attention_matrix.png'.format(i))

    # model.fit(tr_dataset, batch_size=config['batch_size'], epochs=200)
    # model.save_weights('myrun/ViT/' + 'vit_custom')
    return

if __name__ == '__main__':
    main_function()