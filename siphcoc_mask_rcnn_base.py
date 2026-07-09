

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import io
import matplotlib
import numpy as np

from PIL import Image
from six import BytesIO

import orbit
import tensorflow_models as tfm


from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
from official.vision.ops.preprocess_ops import normalize_image
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf


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
from typing import Optional, List, Sequence, Union

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import base_config
from official.vision.configs import common
from official.vision.configs import decoders
from official.vision.configs import backbones

from official.vision.configs import maskrcnn


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])

HIGH = 1280
WIDTH = 1280

COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 600
COCO_VAL_EXAMPLES = 160



def cascadercnn_spinenet_coco_local() -> cfg.ExperimentConfig:
  """COCO object detection with Cascade RCNN-RS with SpineNet backbone."""
  steps_per_epoch = 100
  coco_val_samples = 100
  train_batch_size = 600
  eval_batch_size = 4

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=maskrcnn.MaskRCNNTask(
          # annotation_file=os.path.join(COCO_INPUT_PATH_BASE,
          #                              'instances_val2017.json'),
          model=maskrcnn.MaskRCNN(
              backbone=backbones.Backbone(
                  type='spinenet',
                  spinenet=backbones.SpineNet(
                      model_id='143',
                      min_level=3,
                      max_level=7,
                      stochastic_depth_drop_rate = 0.2
                  )),
              decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
              roi_sampler=maskrcnn.ROISampler(
                               cascade_iou_thresholds=[0.7],
                               foreground_iou_threshold=0.6),
              detection_head=maskrcnn.DetectionHead(
                  num_convs=4,num_fcs=2,num_filters=256,
                  class_agnostic_bbox_pred=True, cascade_class_ensemble=True),
              rpn_head=maskrcnn.RPNHead(
                               # use_separable_conv=True,
                               num_convs=2,
                               num_filters=256),
              mask_head=maskrcnn.MaskHead(
                               num_convs = 4,
                               num_filters=256,
                ),
              anchor=maskrcnn.Anchor(anchor_size=3),
              norm_activation=common.NormActivation(
                  norm_epsilon=0.001,norm_momentum=0.99,
                  use_sync_bn=True, 
                  activation='swish'),
              num_classes=6,
              input_size=[HIGH, WIDTH, 3],
              min_level=3,
              max_level=7,
              include_mask=True),
          losses=maskrcnn.Losses(l2_weight_decay=0.00004),
          train_data=maskrcnn.DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=maskrcnn.Parser(
                  aug_rand_hflip=True, aug_scale_min=1.0, aug_scale_max=1.0)),
          validation_data=maskrcnn.DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          train_steps=steps_per_epoch * 500,
          validation_steps=coco_val_samples // eval_batch_size,
          validation_interval=steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          steps_per_epoch * 475, steps_per_epoch * 490
                      ],
                      'values': [0.32, 0.032, 0.0032],
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
                      'warmup_learning_rate': 0.0067
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
          'task.model.min_level == task.model.backbone.spinenet.min_level',
          'task.model.max_level == task.model.backbone.spinenet.max_level',
      ])
  return config



train_data_input_path = './mydata/trainsrcdata/SIPHCOC/SIPHCOCTrain_m2.tfrecord'
valid_data_input_path = './mydata/trainsrcdata/SIPHCOC/SIPHCOCVerify_m2.tfrecord'
#test_data_input_path = './mydata/BCCD/test-00000-of-00001.tfrecord'
model_dir = './mydata/trainmodels/SIPHCOC/trained_model'
export_dir ='./mydata/trainmodels/SIPHCOC/exported_model'


#CONFIG PART
exp_config = cascadercnn_spinenet_coco_local()
batch_size = 1
num_classes = 11

IMG_SIZE = [HIGH, WIDTH, 3]

# Backbone config.
exp_config.task.freeze_backbone = False
exp_config.task.annotation_file = ''

# Model config.
exp_config.task.model.input_size = IMG_SIZE
exp_config.task.model.num_classes = num_classes + 1
# exp_config.task.model.detection_generator.tflite_post_processing.max_classes_per_detection = exp_config.task.model.num_classes

# Training data config.
exp_config.task.train_data.input_path = train_data_input_path
exp_config.task.train_data.dtype = 'float32'
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.train_data.parser.aug_scale_max = 1.1
exp_config.task.train_data.parser.aug_scale_min = 0.9

# Validation data config.
exp_config.task.validation_data.input_path = valid_data_input_path
exp_config.task.validation_data.dtype = 'float32'
exp_config.task.validation_data.global_batch_size = 1


#TRAINER PART
train_steps = 35940
exp_config.trainer.steps_per_loop = 599 # steps_per_loop = num_of_training_examples // train_batch_size

exp_config.trainer.summary_interval = 599
exp_config.trainer.checkpoint_interval = 599
exp_config.trainer.validation_interval = 599
exp_config.trainer.validation_steps =  90 # validation_steps = num_of_validation_examples // eval_batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 599


exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.18
exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.0015



task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)


if exp_config.runtime.mixed_precision_dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')


# logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]
# distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

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

# model, eval_logs = tfm.core.train_lib.run_experiment(
#     distribution_strategy=distribution_strategy,
#     task=task,
#     mode='train_and_eval',
#     params=exp_config,
#     model_dir=model_dir,
#     run_post_eval=True)


export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[HIGH, WIDTH],
    params=exp_config,
    checkpoint_path= tf.train.latest_checkpoint('./mydata/trainmodels/SIPHCOC/trained_model_599_mask_839') ,#(model_dir),#'./mydata/trained_model_mz/ckpt-1600',
    export_dir=export_dir)


# imported = tf.saved_model.load(export_dir)
# model_fn = imported.signatures['serving_default']
# result = model_fn(image)

