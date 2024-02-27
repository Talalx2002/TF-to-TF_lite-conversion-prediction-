import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='C:\\University\\FYP\\Code\\converted_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image_path = 'C:\\University\\FYP\\Code\\result\\0.jpg'
img = Image.open(image_path).resize((224, 224))  # Resize the image
img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], img_array)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Assuming you have a classification model, you might want to decode the predictions
class_labels = ['0', '1', '2', '3', '4']
predicted_class_index = np.argmax(output_data[0])
predicted_class_name = class_labels[predicted_class_index]

# Display the results
print("Predicted class: ", predicted_class_name)
print("Prediction probabilities: ", output_data[0])

import matplotlib.pyplot as plt

# Prediction probabilities
probabilities = output_data[0]

# Create bar graph
plt.figure(figsize=(10,6))
plt.bar(class_labels, probabilities, color='blue')

# Adding labels
plt.xlabel('Classes')
plt.ylabel('Prediction Probabilities')
plt.title('Prediction Probabilities Bar Graph')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
