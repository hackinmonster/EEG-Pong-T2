import asyncio
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from pylsl import StreamInlet, resolve_stream
import tensorflow as tf
import time


class TrainClassifier:
    def __init__(self, model):
        self.model = model
        self.labels_dict = {}

    def read_file(self, filepath):
        data = []
        with open(filepath, "r") as file:
            csv_reader = csv.reader(file)
            row_num = 0
            for row in csv_reader:
                if row_num > 0:
                    tp9 = np.array(float(row[1])).reshape(-1, 1)
                    tp10 = np.array(float(row[2])).reshape(-1, 1)
                    af7 = np.array(float(row[3])).reshape(-1, 1)
                    af8 = np.array(float(row[4])).reshape(-1, 1)
                    concat = np.concatenate([tp9, af7, af8, tp10], axis=0)
                    data.append(concat)

                else:
                    row_num += 1

        data = np.array(data)
        data = data.reshape(-1, 4)
        data = np.expand_dims(data, axis=0)

        return data

    def read_folder(self, folderpath, label):
        file_paths = os.listdir(folderpath)
        num_files = len(file_paths)

        label = np.array([label])
        labels = np.repeat(label, num_files).reshape(-1, 1)

        data = []

        for file_path in file_paths:
            file_data = self.read_file(os.path.join(folderpath, file_path))
            data.append(file_data)

        data = np.array(data)
        data = data.reshape(-1, file_data.shape[1], file_data.shape[2])

        return data, labels

    def load_dataset_from_directory(self, directory="dataset", batch_size=32, buffer_size=100,
                                    prefetch=tf.data.AUTOTUNE):
        folder_paths = os.listdir(directory)
        dataset = []
        labels = []
        for i, folder_path in enumerate(folder_paths):
            self.labels_dict[i] = folder_path
            folder_path = os.path.join(directory, folder_path)
            data, label = self.read_folder(folder_path, i)
            dataset.append(data)
            labels.append(label)

        dataset = np.array(dataset).reshape(-1, data.shape[1], data.shape[2])
        labels = np.array(labels).reshape(-1, label.shape[1])

        return tf.data.Dataset.from_tensor_slices((dataset, labels)).shuffle(buffer_size).batch(batch_size).prefetch(
            prefetch)

    def decode_prediction(self, prediction):
        prediction = tf.argmax(prediction, axis=1)
        return self.labels_dict[prediction]

    def train_model(self, dataset, model_filepath="model", epochs=1, steps_per_epoch=1000, learning_rate=1e-3,
                    weight_decay=0, metrics=["accuracy"], save=True):
        if model_filepath.endswith(".keras"):
            pass
        else:
            model_filepath = model_filepath + ".keras"

        self.model.compile(loss="SparseCategoricalCrossentropy",
                           optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
                           metrics=metrics)

        self.model.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=steps_per_epoch)

        self.model.summary()

        if save:
            self.model.save(model_filepath)

    def calculate_gradcam_heatmap(self, input_data, layer_name, index, threshold=0):
        grad_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=[self.model.get_layer(layer_name).output,
                                                                              self.model.get_layer("dense").output])

        grad_model.build(input_data.shape)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_data)
            loss = predictions[:, index]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=[0, 1])

        conv_outputs = conv_outputs[0]
        conv_outputs = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.reduce_mean(tf.nn.relu(conv_outputs), axis=-1).numpy()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
        heatmap[heatmap < threshold] = 0
        return heatmap

    def plot_gradcam_heatmap(self, eeg_data, layer_name="conv1d", index=1):
        input_data = eeg_data.squeeze()
        num_channels = input_data.shape[1]
        plt.figure(figsize=(12, 8))

        for i in range(num_channels):
            plt.subplot(num_channels, 1, i + 1)
            plt.plot(input_data[:, i], label=f'Channel {i + 1}', color=f'C{i}')
            plt.title(f'Channel {i + 1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Value')

            heatmap = self.calculate_gradcam_heatmap(eeg_data, layer_name, index)

            img = plt.imshow(np.expand_dims(heatmap, axis=-1).T, cmap='jet', alpha=0.6, aspect='auto',
                             extent=[0, len(input_data), np.min(input_data[:, i]), np.max(input_data[:, i])])
            plt.colorbar(img, ax=plt.gca(), label='Grad-CAM Intensity')

        plt.suptitle("title")
        plt.tight_layout()
        plt.show()


class TestClassifier:
    def __init__(self, model, context_length, num_classes, ema=1):
        self.model = model
        self.maxlen = int(context_length)
        self.ema = ema
        self.eeg_data = tf.random.normal(stddev=40, shape=(self.maxlen, 4))
        self.last_predicted = 0

    def start_stream(self):
        print("Looking for Muse stream.")
        streams = resolve_stream('type', 'EEG')

        if not streams:
            print("No stream found.")
            return None

        self.inlet = StreamInlet(streams[0], max_buflen=1)
        print("Muse found. Starting Test")

    async def eeg_loop(self):
        eeg_data = None

        sample, timestamp = self.inlet.pull_sample(timeout=0.001)

        if sample:
            eeg_data = tf.convert_to_tensor(sample)
            eeg_data = tf.reshape(eeg_data[0:4], (1, 4))
            self.eeg_data = tf.concat([eeg_data, self.eeg_data], axis=0)

        await asyncio.sleep(0.01)

    async def prediction_loop(self):
        self.eeg_data = self.eeg_data[:-1, :]
        prediction = self.model.predict(tf.expand_dims(self.eeg_data, axis=0), verbose=0)
        self.last_predicted = self.ema * prediction + (1 - self.ema) * self.last_predicted

        await asyncio.sleep(0.01)

    async def initialize_async(self):
        eeg_loop = asyncio.create_task(self.eeg_loop())
        model_loop = asyncio.create_task(self.prediction_loop())

        await asyncio.gather(eeg_loop, model_loop)

    async def initialize_loop(self):
        os.system("muselsl stream")
        inlet = self.start_stream()
        print(f"Initializing. This will take roughly {self.maxlen/255}s.")
        for _ in range(self.maxlen):
            await self.initialize_async()
        print("initialization complete")

    async def main_loop(self):
        await self.initialize_async()


class EEGRecordingUtils:
    def start_stream(self):
        print("Looking for Muse stream.")
        eeg_stream = resolve_stream('type', 'EEG')
        ppg_stream = resolve_stream("type", "PPG")

        if not eeg_stream:
            print("No stream found.")
            return None

        self.eeg_inlet = StreamInlet(eeg_stream[0], max_buflen=1)
        print("Muse found. Starting Test")

    def eeg_loop(self):
        eeg_data = None

        eeg_sample, timestamp = self.eeg_inlet.pull_sample(timeout=0.001)

        if eeg_sample:
            eeg_data = tf.convert_to_tensor(eeg_sample)
            eeg_data = tf.reshape(eeg_data[0:4], (1, 4))

        return eeg_data

    def record(self, duration_seconds, file_name):
        duration = duration_seconds * 256
        self.start_stream()
        start_time = time.time()

        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['Timestamp', 'TP9', 'AF7', 'AF8', 'TP10'])

            input("Press enter to Record...")
            #while time.time() - start_time < duration_seconds:
            while duration > 0:
                eeg_data = self.eeg_loop()

                current_time = time.time()

                eeg_flat = eeg_data.numpy().flatten().tolist()

                writer.writerow([current_time] + eeg_flat)

            print(f"Finished recording. Data has been saved to {file_name}")

