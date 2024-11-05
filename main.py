from EEGUtils import EEGRecordingUtils, TrainClassifier, TestClassifier
import tensorflow as tf
import asyncio
import os
import time


def record():

    main_folder_path = "dataset"
    os.makedirs(main_folder_path, exist_ok=True)

    left_folder_path = os.path.join(main_folder_path, "left_jaw_clench")
    right_folder_path = os.path.join(main_folder_path, "right_jaw_clench")
    os.makedirs(left_folder_path, exist_ok=True)
    os.makedirs(right_folder_path, exist_ok=True)

    label = input("Enter 'left_jaw_clench' or 'right_jaw_clench' for the recording label: ").strip().lower()
    if label not in ['left_jaw_clench', 'right_jaw_clench']:
        print("Invalid label. Please enter 'left_jaw_clench' or 'right_jaw_clench'.")
        return

    folder_path = os.path.join("dataset", label)
    os.makedirs(folder_path, exist_ok=True)
    timestamp = int(time.time())
    file_name = f"eeg_recording_{timestamp}.csv"
    file_path = os.path.join(folder_path, file_name)

    duration_seconds = 10

    utils = EEGRecordingUtils()
    utils.start_stream()
    utils.record(duration_seconds, file_path)


def train():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(None, 4)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    trainer = TrainClassifier(model)
    dataset = trainer.load_dataset_from_directory(directory="dataset")
    
    timestamp = int(time.time())
    checkpoint_filepath = f"model_checkpoint_{timestamp}.keras"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True, 
        monitor='loss',        
        save_best_only=True,
        mode='min',
        verbose=1
    )

    trainer.train_model(
        dataset, 
        model_filepath="eeg_model", 
        epochs=10, 
        callbacks=[model_checkpoint_callback] 
    )

    return model, checkpoint_filepath

def evaluate_model(model, test_dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


def test_and_evaluate(model, checkpoint_filepath):
    if tf.io.gfile.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)
        print("Model weights loaded from checkpoint.")

    trainer = TrainClassifier(model)
    
    if not os.path.exists("test_dataset"):
        print("The 'test_dataset' folder does not exist. Please create it.")
        return

    test_dataset = trainer.load_dataset_from_directory(directory="test_dataset")  

    evaluate_model(model, test_dataset)

    tester = TestClassifier(model, context_length=256, num_classes=2)
    asyncio.run(tester.main_loop())


if __name__ == '__main__':

    #10 seconds
    record()

    model, checkpoint_filepath = train()

    test_and_evaluate(model, checkpoint_filepath)

