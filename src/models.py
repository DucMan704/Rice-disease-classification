"""
src/models.py
==============
Builders for the 5 backbone architectures. Every builder returns:
    (keras_model, base_model)
so the training code can freeze/unfreeze `base_model` independently of the
full model (which also includes the preprocessing + classification head).

All 5 architectures are shipped with tf.keras.applications, so no extra
pip installs are required on Kaggle.
"""

import config


def _classification_head(base_output, num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    x = layers.GlobalAveragePooling2D(name="gap")(base_output)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(config.DROPOUT_RATE / 2, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    return outputs


def _build(base_ctor, preprocess_fn, num_classes, img_size):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_image")
    x = layers.Lambda(preprocess_fn, name="preprocess")(inputs)

    base_model = base_ctor(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False  # phase 1: frozen backbone

    outputs = _classification_head(base_model.output, num_classes)
    model = keras.Model(inputs, outputs, name=base_model.name + "_classifier")
    return model, base_model


def build_efficientnet_b0(num_classes, img_size=config.IMG_SIZE):
    from tensorflow.keras.applications import efficientnet
    return _build(efficientnet.EfficientNetB0, efficientnet.preprocess_input, num_classes, img_size)


def build_resnet50(num_classes, img_size=config.IMG_SIZE):
    from tensorflow.keras.applications import resnet50
    return _build(resnet50.ResNet50, resnet50.preprocess_input, num_classes, img_size)


def build_densenet121(num_classes, img_size=config.IMG_SIZE):
    from tensorflow.keras.applications import densenet
    return _build(densenet.DenseNet121, densenet.preprocess_input, num_classes, img_size)


def build_mobilenet_v3_large(num_classes, img_size=config.IMG_SIZE):
    from tensorflow.keras.applications import MobileNetV3Large
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    return _build(MobileNetV3Large, preprocess_input, num_classes, img_size)


def build_convnext_tiny(num_classes, img_size=config.IMG_SIZE):
    from tensorflow.keras.applications import convnext
    return _build(convnext.ConvNeXtTiny, convnext.preprocess_input, num_classes, img_size)


MODEL_BUILDERS = {
    "EfficientNetB0": build_efficientnet_b0,
    "ResNet50": build_resnet50,
    "DenseNet121": build_densenet121,
    "MobileNetV3Large": build_mobilenet_v3_large,
    "ConvNeXtTiny": build_convnext_tiny,
}


def unfreeze_top_fraction(base_model, fraction=config.FINE_TUNE_UNFREEZE_FRACTION):
    """Unfreeze the last `fraction` of layers in base_model for fine-tuning.
    BatchNorm layers are kept frozen (standard practice) to avoid destroying
    running statistics learned on ImageNet.
    """
    base_model.trainable = True
    n_layers = len(base_model.layers)
    n_unfreeze = max(1, int(n_layers * fraction))
    freeze_until = n_layers - n_unfreeze

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # keep normalization layers frozen even inside the unfrozen block
            cls_name = layer.__class__.__name__
            if "BatchNormalization" in cls_name:
                layer.trainable = False
            else:
                layer.trainable = True
    return n_unfreeze, n_layers
