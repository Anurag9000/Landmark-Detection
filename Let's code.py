import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5

# Function to normalize landmarks (original size = 178x218)
def normalize_landmarks(landmarks):
    return tf.stack([
        landmarks['left_eye'][0] / 178.0, landmarks['left_eye'][1] / 218.0,
        landmarks['right_eye'][0] / 178.0, landmarks['right_eye'][1] / 218.0,
        landmarks['nose'][0] / 178.0, landmarks['nose'][1] / 218.0,
        landmarks['mouth_left'][0] / 178.0, landmarks['mouth_left'][1] / 218.0,
        landmarks['mouth_right'][0] / 178.0, landmarks['mouth_right'][1] / 218.0
    ])

# Preprocessing function for tf.data
def preprocess(sample):
    image = tf.image.resize(sample['image'], (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    landmarks = normalize_landmarks(sample['landmarks'])
    return image, landmarks

# Load and prepare dataset
dataset = tfds.load('celeb_a', split='train[:10000]', shuffle_files=True)
dataset = dataset.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tfds.load('celeb_a', split='train[10000:10200]')
val_dataset = val_dataset.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10)  # 5 landmarks * (x, y)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(dataset, validation_data=val_dataset, epochs=EPOCHS)

# Predict from validation dataset
sample_images, sample_labels = next(iter(val_dataset))
predictions = model.predict(sample_images)

# Visualize predictions
def show_predictions(images, true_landmarks, pred_landmarks, count=5):
    plt.figure(figsize=(15, 5))
    for i in range(count):
        img = images[i].numpy()
        true_pts = (true_landmarks[i].numpy().reshape(-1, 2) * [178, 218]).astype(int)
        pred_pts = (pred_landmarks[i].reshape(-1, 2) * [178, 218]).astype(int)
        img_resized = tf.image.resize(img, (218, 178)).numpy()

        plt.subplot(1, count, i + 1)
        plt.imshow(img_resized)
        for x, y in true_pts:
            plt.plot(x, y, 'go')  # green: ground truth
        for x, y in pred_pts:
            plt.plot(x, y, 'rx')  # red: prediction
        plt.axis('off')
    plt.show()

show_predictions(sample_images, sample_labels, predictions)
