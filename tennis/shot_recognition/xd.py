import tensorflow as tf

try:
    interpreter = tf.lite.Interpreter(model_path="movenet.tflite")
    interpreter.allocate_tensors()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
