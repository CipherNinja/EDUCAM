import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
base_path = r"C:\Users\Priyesh Pandey\OneDrive\Desktop\EDUCAM\ML\Dataset\Train\Human Faces Dataset"
ai_path = os.path.join(base_path, "AI-Generated Images")
real_path = os.path.join(base_path, "Real Images")  # Adjust if real images are elsewhere

# Load image paths and labels
def load_dataset():
    ai_images = [os.path.join(ai_path, fname) for fname in os.listdir(ai_path) if fname.endswith('.jpg')]
    real_images = [os.path.join(real_path, fname) for fname in os.listdir(real_path) if fname.endswith('.jpg')]
    
    images = ai_images + real_images
    labels = [0] * len(ai_images) + [1] * len(real_images)  # 0: AI, 1: Real
    return images, labels

images, labels = load_dataset()

# Split dataset (80/20)
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Image data generators for loading images in batches
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Function to create a dataframe for flow_from_dataframe
import pandas as pd
train_df = pd.DataFrame({'filename': train_images, 'class': [str(label) for label in train_labels]})
test_df = pd.DataFrame({'filename': test_images, 'class': [str(label) for label in test_labels]})

# Data generators
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='class',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Train model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust based on convergence
    validation_data=test_generator
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Save model
model.save('human_faces_cnn_model.h5')