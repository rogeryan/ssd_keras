import os
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.data_augmentation_chain_variable_input_size import DataAugmentationVariableInputSize
from data_generator.data_augmentation_chain_constant_input_size import DataAugmentationConstantInputSize
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation

# 日志配置，便于记录训练过程信息


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


init_log_config()

# 1. Set the model configuration parameters
img_height = 300  # Height of the input images
img_width = 480  # Width of the input images
img_channels = 3  # Number of color channels of the input images
# Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_mean = 127.5
# Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5
n_classes = 5  # Number of positive classes
# An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.08, 0.16, 0.32, 0.64, 0.96]
# The list of aspect ratios for the anchor boxes
aspect_ratios = [0.5, 1.0, 2.0]
# Whether or not you want to generate two anchor boxes for aspect ratio 1
two_boxes_for_ar1 = True
steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
clip_boxes = False
# The list of variances by which the encoded target coordinates are scaled
variances = [1.0, 1.0, 1.0, 1.0]
# Whether or not the model is supposed to use coordinates relative to the image size
normalize_coords = True
model_saved = 'ssd7.h5'


# 加载之前保存的模型或者创建新模型
K.clear_session()  # Clear previous models from memory.
if os.path.exists(model_saved):
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model = load_model(model_saved, custom_objects={
                       'AnchorBoxes': AnchorBoxes, 'compute_loss': ssd_loss.compute_loss})
else:
    model = build_model(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_global=aspect_ratios,
                        aspect_ratios_per_layer=None,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=intensity_mean,
                        divide_by_stddev=intensity_range)

    # 2: Optional: Load some weights
    #model.load_weights('./ssd7_weights.h5', by_name=True)
    # 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


train_dataset = DataGenerator(
    load_images_into_memory=False, hdf5_dataset_path='dataset_udacity_traffic_train.h5')
val_dataset = DataGenerator(
    load_images_into_memory=False, hdf5_dataset_path='dataset_udacity_traffic_val.h5')
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

logger.info("Number of images in the training dataset:\t{:>6}".format(
    train_dataset_size))
logger.info("Number of images in the validation dataset:\t{:>6}".format(
    val_dataset_size))


# part 3
# 3: Set the batch size.
batch_size = 16
# 4: Define the image processing chain.
data_augmentation_chain = DataAugmentationConstantInputSize(
    random_brightness=(-48, 48, 0.5),
    random_contrast=(0.5, 1.8, 0.5),
    random_saturation=(0.5, 1.8, 0.5),
    random_hue=(18, 0.5),
    random_flip=0.5,
    random_translate=((0.03, 0.5), (0.03, 0.5), 0.5),
    random_scale=(0.5, 2.0, 0.5),
    n_trials_max=3,
    clip_boxes=True,
    overlap_criterion='area',
    bounds_box_filter=(0.3, 1.0),
    bounds_validator=(0.5, 1.0),
    n_boxes_min=1,
    background=(0, 0, 0))
# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation_chain],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# part 4
# Define model callbacks.
# TODO: Set the filepath under which you want to save the weights.
model_checkpoint = ModelCheckpoint(filepath='ssd7_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='ssd7_training_log.csv',
                       separator=',',
                       append=True)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=1)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=1,
                                         epsilon=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]

# TODO: Set the epochs to train for.
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 20
steps_per_epoch = 1000

## test
tr = iter(train_generator)
tr0 = next(tr)
print(type(tr0))
for x in tr0:
    logger.info(x.shape)
    logger.info('='*30)


history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=final_epoch,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=ceil(val_dataset_size/batch_size),
                    initial_epoch=initial_epoch)

plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right', prop={'size': 24})