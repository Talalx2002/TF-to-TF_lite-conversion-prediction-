from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the Keras model
model = load_model('Downloads/DR_model_15_19(99.604).h5')

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Specify the path to save the TensorFlow Lite model
tflite_model_path = r'C:\University\FYP\Code\converted_model.tflite'

# Save the TensorFlow Lite model to file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TensorFlow Lite model saved to: {tflite_model_path}')