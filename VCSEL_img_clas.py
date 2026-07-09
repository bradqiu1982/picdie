
import os
import io
import matplotlib
import numpy as np

import pathlib
import random

from PIL import Image
from six import BytesIO

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds
from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib


from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder

from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling import performance
from official.vision import registry_imports  # pylint: disable=unused-import
# from official.vision.utils import summary_manager


import dataclasses
import os
from typing import List, Optional, Tuple, Union, Sequence

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones

from official.vision.configs import image_classification


HIGH = 480
WIDTH = 480

IMAGENET_TRAIN_EXAMPLES = 800
IMAGENET_VAL_EXAMPLES = 120
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


@dataclasses.dataclass
class DataConfig_local(cfg.DataConfig):
  """Input config for training."""
  input_path: Union[Sequence[str], str, hyperparams.Config] = ''
  weights: Optional[hyperparams.base_config.Config] = None
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  is_multilabel: bool = False
  aug_rand_hflip: bool = False
  aug_crop: Optional[bool] = False
  crop_area_range: Optional[Tuple[float, float]] = (1.0, 1.0)
  aug_type: Optional[
      common.Augmentation] = None  # Choose from AutoAugment and RandAugment.
  color_jitter: float = 0.
  random_erasing: Optional[common.RandomErasing] = None
  file_type: str = 'tfrecord'
  image_field_key: str = 'image/encoded'
  label_field_key: str = 'image/class/label'
  decode_jpeg_only: bool = True
  mixup_and_cutmix: Optional[common.MixupAndCutmix] = None
  decoder: Optional[common.DataDecoder] = common.DataDecoder()

  # Keep for backward compatibility.
  aug_policy: Optional[str] = None  # None, 'autoaug', or 'randaug'.
  randaug_magnitude: Optional[int] = 1


def image_classification_imagenet_resnetrs_local() -> cfg.ExperimentConfig:
  """Image classification on imagenet with resnet-rs."""
  train_batch_size = 8
  eval_batch_size = 8
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
  config = cfg.ExperimentConfig(
      task=image_classification.ImageClassificationTask(
          model=image_classification.ImageClassificationModel(
              num_classes=10,
              input_size=[HIGH, WIDTH, 3],
              backbone=backbones.Backbone(
                  type='resnet',
                  resnet=backbones.ResNet(
                      model_id=152,
                      stem_type='v1',
                      resnetd_shortcut=True,
                      replace_stem_max_pool=True,
                      se_ratio=0.25,
                      stochastic_depth_drop_rate=0.0)),
              dropout_rate=0.2,
              norm_activation=common.NormActivation(
                  norm_momentum=0.0,
                  norm_epsilon=1e-5,
                  use_sync_bn=False,
                  activation='swish')),
          losses=image_classification.Losses(l2_weight_decay=4e-5, label_smoothing=0.1),
          train_data=DataConfig_local(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              # aug_type=common.Augmentation(
              #     type='randaug', randaug=common.RandAugment(magnitude=10))
              ),
          validation_data=DataConfig_local(
              input_path=os.path.join(IMAGENET_INPUT_PATH_BASE, 'valid*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=350 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'ema': {
                  'average_decay': 0.9999,
                  'trainable_weights_only': False,
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 1.6,
                      'decay_steps': 350 * steps_per_epoch
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
  return config




model_dir = './mydata/trainmodels/VCLAS_DATA/trained_model_imgcls/'
export_dir ='./mydata/trainmodels/VCLAS_DATA/exported_model_imgcls/'

train_data_input_path = './mydata/trainsrcdata/VCLAS_DATA/train_dataset.tfrecord'
valid_data_input_path = './mydata/trainsrcdata/VCLAS_DATA/valid_dataset.tfrecord'

classcnt = 5
exp_config = image_classification_imagenet_resnetrs_local()
IMG_SIZE = [HIGH, WIDTH, 3]
batch_size = 4


exp_config.task.freeze_backbone = False


exp_config.task.model.num_classes = classcnt
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.backbone.resnet.model_id = 152


# Training data config.
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
# exp_config.task.train_data.parser.aug_scale_max = 1.0
# exp_config.task.train_data.parser.aug_scale_min = 1.0

# Validation data config.
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = batch_size

#steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size
#train_steps=350 * steps_per_epoch 'decay_steps': 350 * steps_per_epoch 'warmup_steps': 5 * steps_per_epoch
#steps_per_loop=steps_per_epoch summary_interval=steps_per_epoch checkpoint_interval=steps_per_epoch
#validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size  validation_interval=steps_per_epoch

train_steps=8000
exp_config.trainer.steps_per_loop = 50
exp_config.trainer.summary_interval = 50
exp_config.trainer.checkpoint_interval = 50
exp_config.trainer.validation_interval = 50
exp_config.trainer.validation_steps =  50
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100



if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')


# logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]
# if 'GPU' in ''.join(logical_device_names):
#   distribution_strategy = tf.distribute.MirroredStrategy()
# elif 'TPU' in ''.join(logical_device_names):
#   tf.tpu.experimental.initialize_tpu_system()
#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
#   distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# else:
#   print('Warning: this will be really slow.')
#   distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

distribution_strategy = tf.distribute.OneDeviceStrategy('/device:GPU:0')


task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)


# for images, labels in task.build_inputs(exp_config.task.train_data).take(1):
#   print()
#   print(f'images.shape: {str(images.shape):16}  images.dtype: {images.dtype!r}')
#   print(f'labels.shape: {str(labels.shape):16}  labels.dtype: {labels.dtype!r}')
#   for idx in range(0,8):
#     imgs = images.numpy()
#     min = imgs.min()
#     max = imgs.max()
#     delta = max - min
#     img = ((imgs[idx]-min)/delta)*255.0
#     print(img)
#     img = img.astype(np.uint8)
#     im = Image.fromarray(img)
#     im.save('./cls_img/sample'+str(idx)+'_'+str(labels[idx].numpy())+'.jpg')
#     idx = idx + 1


model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train_and_eval',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True)


# export_saved_model_lib.export_inference_graph(
#     input_type='image_tensor',
#     batch_size=1,
#     input_image_size=[HIGH, WIDTH],
#     params=exp_config,
#     checkpoint_path=tf.train.latest_checkpoint(model_dir),
#     export_dir=export_dir)


# imported = tf.saved_model.load(export_dir)
# model_fn = imported.signatures['serving_default']

