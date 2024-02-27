# TF to TFLite Conversion and Prediction

This repository provides scripts for converting TensorFlow models to TensorFlow Lite format and conducting predictions using the converted models. It's designed for deploying machine learning models on resource-limited devices and edge environments.

## Usage

1. **Model Conversion**:
   - Use `TF_to_TFLite.py` to convert your TensorFlow model to TensorFlow Lite format.
   - Modify the file paths in the script according to your model's location.
   - Run the script using the following command:
     ```bash
     python TF_to_TFLite.py
     ```

2. **Prediction**:
   - Use `TFLite_Prediction.py` to perform predictions using the converted TensorFlow Lite model.
   - Ensure the TensorFlow Lite model is correctly referenced in the script.
   - Run the script using the following command:
     ```bash
     python TFLite_Prediction.py
     ```

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pillow
- Matplotlib

## Folder Structure

- **Conversion and Prediction**: Contains the Python scripts for model conversion and prediction.
  - `TF to TFLite.py`: Script to convert a TensorFlow model to TensorFlow Lite format.
  - `TFLite Prediction.py`: Script to perform predictions using the converted TensorFlow Lite model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
