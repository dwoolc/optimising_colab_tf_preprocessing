# copy in a zip folder from GDrive, unzip and delete

!cp '/content/drive/My Drive/data.zip' '/content/'
!unzip '/content/data.zip' -d '/content/'
!rm '/content/data.zip'

# data directories
train_dir = '/content/data/train'
test_dir = '/content/data/test'
valid_dir = '/content/data/validate'

# using a custom version of image_dataset_from_directory
# returns "return dataset, labels, image_paths"
train_dataset, train_labels, train_paths = image_dataset_from_directory(train_dir,
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             seed = 1,
                                             image_size=IMG_SIZE)
                                             
BATCH_SIZE = 64

IMG_SIZE = (224, 224)

# set datasets
train_dataset, train_labels, train_paths = image_dataset_from_directory(train_dir,
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             seed = 1,
                                             image_size=IMG_SIZE)

test_dataset, test_labels, test_paths = image_dataset_from_directory(test_dir,
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validate_dataset, validate_labels, validate_paths = image_dataset_from_directory(valid_dir,
                                             shuffle=False,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

class_names = train_dataset.class_names

# add prefetch with autotune
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# import cnn specific preprocess (mobnet)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# map preprocess to dataset
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y))

# cache if we can. Makes us >>> faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

IMG_SHAPE = IMG_SIZE + (3,)

# compile model
base_model = tf.keras.applications.tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
                                             
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names),activation='softmax')

inputs = tf.keras.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)

base_learning_rate = 0.0001
model_fe.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# train model
epoch_num = 50

history = model.fit(x=train_features, y=train_labels,
                    validation_data=(test_features, test_labels),
                    batch_size=BATCH_SIZE,
                    epochs=epoch_num)
