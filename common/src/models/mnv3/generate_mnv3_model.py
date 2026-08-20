#!/usr/bin/env python3
# Copyright 2026 The CFU-Playground Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import tensorflow as tf

def generate_mobilenetv3_minimalistic():
    print("==================================================")
    print("Generating MobileNetV3-Small Minimalistic INT8 Model")
    print("==================================================")

    # 1. Model Parameters
    INPUT_SHAPE = (160, 160, 3)
    ALPHA = 0.35
    NUM_CLASSES = 2
    
    # Output paths
    output_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(output_dir, "model_mobilenetv3_small_min.tflite")
    dat_path = os.path.join(output_dir, "input_00001.dat")

    # 2. Build MobileNetV3 Minimalistic Keras Model
    # minimalistic=True disables SE modules & uses ReLU6 activations
    print("Instantiating MobileNetV3-Small (minimalistic=True)...")
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=INPUT_SHAPE,
        alpha=ALPHA,
        minimalistic=True,
        include_top=True,
        weights=None,
        classes=NUM_CLASSES
    )

    # 3. Calibration Generator for PTQ
    def representative_dataset_gen():
        for _ in range(50):
            data = np.random.uniform(-1.0, 1.0, size=(1, *INPUT_SHAPE)).astype(np.float32)
            yield [data]

    # 4. TFLite INT8 Quantization
    print("Quantizing model to INT8 flatbuffer...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()

    # 5. Save .tflite flatbuffer
    with open(model_path, "wb") as f:
        f.write(tflite_model_quant)
    print(f"Saved TFLite flatbuffer to: {model_path} ({len(tflite_model_quant)} bytes)")

    # 6. Generate Golden Test Vector (.dat)
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sample_input = np.random.randint(-128, 127, size=input_details[0]['shape'], dtype=np.int8)
    sample_input.tofile(dat_path)

    interpreter.set_tensor(input_details[0]['index'], sample_input)
    interpreter.invoke()
    expected_output = interpreter.get_tensor(output_details[0]['index'])

    print(f"Saved test vector input to: {dat_path}")
    print(f"Expected Output Logits: {expected_output.flatten()}")
    print("Done!")

if __name__ == "__main__":
    generate_mobilenetv3_minimalistic()
