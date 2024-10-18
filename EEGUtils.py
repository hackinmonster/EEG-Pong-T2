import csv
import os
import numpy as np
import tensorflow as tf


class TrainUtils:
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

    def load_dataset_from_directory(self, directory, batch_size=32, buffer_size=100, prefetch=tf.data.AUTOTUNE):
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
                    weight_decay=0, metrics=["accuracy"]):
        if model_filepath.endswith(".keras"):
            pass
        else:
            model_filepath = model_filepath + ".keras"

        self.model.compile(loss="SparseCategoricalCrossentropy",
                           optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
                           metrics=metrics)
        self.model.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=steps_per_epoch)
        self.model.save(model_filepath)


class RecordingUtils:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def struct_create(self, number_of_classes):
        os.mkdir(self.dataset_path)
        for i in range(number_of_classes):
            user_response = input(f"What is the name of class {i+1}?")
            os.mkdir(self.dataset_path, user_response)

        print("Dataset Structured. Please proceed.")

    def stream_eeg_data(self):
        os.system("muselsl stream")

    def view_eeg_data(self, version=2):
        os.system(f"muselsl view --version {version}")

    def record_eeg_data(self, duration):
        duration = duration + 5
        os.system(f"muselsl record --duration {duration}")

