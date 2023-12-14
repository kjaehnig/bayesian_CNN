import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import scipy as sp
import seaborn as sns
import os
# import bcnn_utils as bcu
import cv2
import tqdm

sns.set_style('whitegrid')
sns.set_context('talk')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D,
                                     Flatten, Dropout, DepthwiseConv2D,
                                     Activation, BatchNormalization, SpatialDropout2D)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import keras_tuner as kt
from keras import backend as K
from tensorflow.keras.saving import get_custom_objects

from sklearn.utils import class_weight


def get_convolutional_reparameterization_layer(input_shape=None, divergence_fn=None, filters=16,name=None):
    """
    This function should create an instance of a Convolution2DReparameterization
    layer according to the above specification.
    The function takes the input_shape and divergence_fn as arguments, which should
    be used to define the layer.
    Your function should then return the layer instance.
    """
    if input_shape == None:
        layer = tfpl.Convolution2DReparameterization(
            filters=filters, kernel_size=(3, 3),padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            name=name
            )
    if input_shape != None:
        layer = tfpl.Convolution2DReparameterization(
            input_shape=input_shape, filters=32, kernel_size=(3, 3),
            padding='same',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence,
            name=name
            )
    return layer

def get_dense_reparameterization_layer(divergence_fn=None, name=None):

    """

    """

    layer = tfpl.DenseReparameterization(
        units=tfpl.OneHotCategorical.params_size(6),
        activation=None,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        kernel_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
        bias_divergence_fn=divergence_fn,
        name=name
        )

    return layer

def get_posterior(kernel_size, bias_size, dtype=None):
    """
    This function should create the posterior distribution as specified above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the posterior distribution.
    """
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])

def spike_and_slab(event_shape, dtype):
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.5, 0.5]),
        components=[
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=1.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1),
            tfd.Independent(tfd.Normal(
                loc=tf.zeros(event_shape, dtype=dtype),
                scale=10.0*tf.ones(event_shape, dtype=dtype)),
                            reinterpreted_batch_ndims=1)],
    name='spike_and_slab')
    return distribution

def get_prior(kernel_size, bias_size, dtype=None):
    """
    This function should create the prior distribution, consisting of the
    "spike and slab" distribution that is described above.
    The distribution should be created using the kernel_size, bias_size and dtype
    function arguments above.
    The function should then return a callable, that returns the prior distribution.
    """
    n = kernel_size+bias_size
    prior_model = Sequential([tfpl.DistributionLambda(lambda t : spike_and_slab(n, dtype))])
    return prior_model

bs = 64
def img_gen(trdir, tsdir):

    trgen = tf.keras.utils.image_dataset_from_directory(
        trdir,
        image_size=(150, 150),
        batch_size=bs,
        label_mode='categorical',
        shuffle=True)


    tsgen = tf.keras.utils.image_dataset_from_directory(
        tsdir,
        image_size=(150, 150),
        batch_size=bs,
        label_mode='categorical',
        shuffle=True)

    return trgen, tsgen


def save_bnn_weights(model, model_name, save_path):
    import h5py
    file = h5py.File(f'{save_path}{model_name}.h5', 'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight' + str(i), data=weight[i])
    file.close()
    print("..Done..")
    return

def load_bnn_weights(model, model_name, save_path):
    file = h5py.File(f'{save_path}{model_name}.h5', 'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)
    return model

def plot_model_performance(bmdlhist):
    fig,ax = plt.subplots(figsize=(10,10), nrows=2)
    ax[0].plot(bmdlhist.history['val_accuracy'],
               label='val_accuracy',marker='o',ms=3)
    ax[0].plot(bmdlhist.history['accuracy'],
               label='accuracy', marker='o',ms=3)
    ax[0].set_title("model accuracy")
    ax[0].set_ylabel("accuracy")
    ax[0].legend()

    ax[1].plot(bmdlhist.history['val_loss'],
               label='val_loss',marker='o',ms=3)
    ax[1].plot(bmdlhist.history['loss'],
               label='loss',marker='o',ms=3)
    ax[1].set_title("model loss")
    ax[1].set_ylabel('loss')
    ax[1].legend()

    plt.savefig("/mnt/g/WSL/wsl_ml_figures/intel_bayes_model_hist.png",bbox_inches='tight',dpi=150)
    plt.close()

def generate_labels_and_preds(mdl, tsgen, return_data=False):
    """
    This function corrects not using a seed with the Image Data Generators
    when trying to compute validation metrics or confusion matricies. It
    simply computes predictions from generator batches, as well as the batch
    predictions, returning one-to-one arrays of truths and predictions.
    """
    from tqdm import tqdm

    lbls = np.array([])
    prds = np.array([])

    xarr = np.empty(shape=(0,150,150,3))
    yarr = np.empty(shape=(0,6))
    btch_cnt = 0
    for btch in tqdm(tsgen.as_numpy_iterator(), total=int(3000/64)):
         blbl = np.argmax(btch[1],axis=1)
         bprd = np.argmax(mdl.predict(btch[0],verbose=0),axis=1)
         # if btch_cnt >= 90:
             # print(bprd, blbl)
         lbls = np.hstack([lbls, blbl])
         prds = np.hstack([prds, bprd])

         btch_cnt += 1

         if return_data:
            xarr = np.vstack([xarr,btch[0]])
            yarr = np.vstack([yarr,btch[1]])
    if return_data:
        return (lbls,prds,xarr,yarr)
    else:
        return lbls, prds

def get_correct_indices(model, x, labels):
    preds = np.empty(shape=(0,6))
    for ii in range(int(x.shape[0]/30)):
        preds = np.vstack([preds, model(x[ii*30:ii*30+30]).mean().numpy()])

    # y_model = model(x)
    correct = np.argmax(preds,axis=1) == np.squeeze(labels)
    correct_indices = [i for i in range(x.shape[0]) if correct[i]]
    incorrect_indices = [i for i in range(x.shape[0]) if not correct[i]]
    return correct_indices, incorrect_indices

def plot_entropy_distribution(model, x, labels):
    probs = np.empty(shape=(0,6))
    for ii in range(int(x.shape[0]/30)):
        probs = np.vstack([probs, model(x[ii*30:ii*30+30]).mean().numpy()])

    # probs = model(x).mean().numpy()
    entropy = -np.sum(probs * np.log(probs), axis=1)

    fig,axes = plt.subplots(1,2, figsize=(10,4))
    fig.subplots_adjust(wspace=0.001, hspace=None)
    axes[1].tick_params(labelleft=False)

    for i,category in zip(range(2),['correct','incorrect']):
        entropy_category = entropy[get_correct_indices(model,x,labels)[i]]
        mean_entropy = np.mean(entropy_category[np.isfinite(entropy_category)])
        num_samples = entropy_category.shape[0]
        title = f"{category}ly labeled ({num_samples/x.shape[0]*100:.1f})"
        axes[i].hist(entropy_category, weights=(1/num_samples)*np.ones(num_samples))
        axes[i].annotate(f"Mean: {mean_entropy:.3f} bits", (0.4,0.9), ha='center')
        axes[i].set_xlabel('entropy (bits)')
        axes[i].set_ylim([0,1])
        axes[i].set_ylabel('probability' if category is 'correct' else '')
        axes[i].set_title(title)


    plt.show()

def plot_per_class_entropy_distributions(model, x, labels, class_names):
    sns.set_style('white')
    sns.set_context('talk')
    probs = np.empty(shape=(0,6))
    for ii in range(int(x.shape[0]/30)):
        probs = np.vstack([probs, model(x[ii*30:ii*30+30]).mean().numpy()])
    y_model = probs.argmax(axis=1)
    entropy = -np.sum(probs*np.log(probs), axis=1)

    corr_dict = {}
    for ii in class_names: corr_dict[ii] = []

    incorr_dict = {}
    for ii in class_names: incorr_dict[ii] = []

    for ii in range(x.shape[0]):
        true_lbl = class_names[np.squeeze(labels[ii]).astype('int')]

        correct_pred = class_names[y_model[ii]] == true_lbl

        if correct_pred and np.isfinite(entropy[ii]):
            corr_dict[true_lbl].append(entropy[ii])
        if not correct_pred and np.isfinite(entropy[ii]):
            incorr_dict[true_lbl].append(entropy[ii])

    fig, axs = plt.subplots(1, 2, figsize=(15,7))
    axs[1].tick_params(labelleft=False)

    corr_seq = [np.array(corr_dict[ii]) for ii in corr_dict.keys()]
    incorr_seq = [np.array(incorr_dict[ii]) for ii in incorr_dict.keys()]

    axs[0].boxplot(corr_seq,showmeans=False,showfliers=False)
    axs[1].boxplot(incorr_seq,showmeans=False,showfliers=False)

    axs[0].tick_params(
        labelsize=14)
    axs[1].tick_params(
        labelsize=14)
    #
    # inset = axs[0].inset_axes([0,0.5,1,0.45])
    # inset.boxplot(corr_seq,showmeans=False, showfliers=False)
    # inset.tick_params(
    #     left=False,
    #     right=True,
    #     labelleft=False,
    #     labelright=True,
    #     labelsize=8)
    # inset.text(1, 0.015, 'Zoom-in', fontsize=14,ha='left',va='top')

    labels = [item.get_text() for item in axs[0].get_xticklabels()]


    axs[0].set_ylim(axs[1].set_ylim())
    axs[0].set_ylabel("entropy [bits]")
    axs[0].set_xlabel("class labels")
    axs[1].set_xlabel("class labels")

    axs[0].set_xticklabels([str(ii) for ii in class_names], rotation=45)
    axs[1].set_xticklabels([str(ii) for ii in class_names], rotation=45)
    # inset.set_xticklabels([str(ii) for ii in class_names])

    axs[0].set_title('Correctly classified')
    axs[1].set_title('Incorrectly classified')
    plt.show()

def grab_predict_random_image(bmodel, imggen):
    sns.set_style('white')
    n_iter = 300
    # generate random image with imggen

    num_classes = len(imggen.class_names)

    rand_id = str(np.random.randint(low=1,high=9999,size=1).squeeze()).zfill(4)

    for ii in imggen.as_numpy_iterator():
        randbatch = ii
        break
    batch_size = bs
    randint = np.random.randint(low=0, high=batch_size - 1)

    image = randbatch[0][randint]
    true_label = np.argmax(randbatch[1][randint])

    class_labels = imggen.class_names

    fig = plt.figure(figsize=(20, 5))
    spec = GridSpec(nrows=25, ncols=100, figure=fig)

    imgax = fig.add_subplot(spec[:, :25])
    barax = fig.add_subplot(spec[:, 30:])
    # read image
    # img = cv2.imread(image)
    img = cv2.cvtColor(image/255., cv2.COLOR_BGR2RGB)

    # show the image
    imgax.imshow(img)
    imgax.axis('off')
    imgax.set_title(f'actual label: {class_labels[true_label]}')
    # img_resize = (cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_CUBIC))/255.

    predicted_probabilities = np.empty(shape=(n_iter, num_classes))

    for i in range(n_iter):
        predicted_probabilities[i] = bmodel(image[np.newaxis, :]).mean().numpy()[0]
    # print(predicted_probabilities)
    pct_2p5 = np.array([np.percentile(predicted_probabilities[:, i], 2.5) for i in range(num_classes)])
    pct_97p5 = np.array([np.percentile(predicted_probabilities[:, i], 97.5) for i in range(num_classes)])

    pct_50 = np.array([np.percentile(predicted_probabilities[:, i], 50) for i in range(num_classes)])

    pred_label = np.argmax(pct_50)
    # mdl_pred = bmodel.predict(img)
    # print(mdl_pred)
    # fig, ax = plt.subplots(figsize=(12, 6))
    bar = barax.bar(np.arange(num_classes), pct_97p5, color='red')
    bar[true_label].set_color('green')
    barax.bar(np.arange(num_classes), pct_2p5 - 0.02, lw=2, color='white')
    barax.set_xticklabels([''] + [x for x in class_labels])
    barax.set_ylim([0, 1])
    barax.set_ylabel('Probability')
    barax.set_title(f'50p model pred: {class_labels[np.argmax(pct_50)]}')

    str_per = str(np.round(max(pct_50),3)*100)[:4].replace('.','_')

    plt.savefig(f"/mnt/g/WSL/wsl_ml_figures/intel_img_preds/intel_bnn_lbl_{class_labels[true_label]}_prd_{class_labels[np.argmax(pct_50)]}_{str_per}per_{rand_id}.png", bbox_inches='tight', dpi=150)
    plt.close()
    # plt.show()

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  import matplotlib.pyplot as plt
  import itertools
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 4
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure
