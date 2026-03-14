import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import os
import pickle
from sklearn.metrics import confusion_matrix

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

MODEL_PATH = "cancer_cell_model.keras"
DATA_INFO_PATH = "data_info.pkl"


# ------------------ Metrics ------------------ #

def sensitivity(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pos + K.epsilon())


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1-y_true)*(1-y_pred), 0, 1)))
    neg = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return tn / (neg + K.epsilon())


# ------------------ Image Processing ------------------ #

def load_and_preprocess_image(img_path):

    try:
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        return img_array

    except Exception as e:
        print("Image loading error:", e)
        return None


# ------------------ Dataset Loader ------------------ #

def load_dataset(data_dir):

    images = []
    labels_int = []

    class_names = sorted(os.listdir(data_dir))

    for class_index, class_name in enumerate(class_names):

        class_dir = os.path.join(data_dir, class_name)

        if os.path.isdir(class_dir):

            for img_file in os.listdir(class_dir):

                if img_file.lower().endswith((".png",".jpg",".jpeg",".webp")):

                    img_path = os.path.join(class_dir, img_file)

                    img_array = load_and_preprocess_image(img_path)

                    if img_array is not None:
                        images.append(img_array)
                        labels_int.append(class_index)

    if len(images) == 0:
        print("No images found in dataset")
        return np.array([]), np.array([]), class_names

    images = np.vstack(images)

    labels = tf.keras.utils.to_categorical(
        labels_int,
        num_classes=len(class_names)
    )

    return images, labels, class_names


# ------------------ Save Dataset Info ------------------ #

def save_data_info(data_dir, class_names):

    data_info = {
        "data_dir": data_dir,
        "class_names": class_names
    }

    with open(DATA_INFO_PATH,"wb") as f:
        pickle.dump(data_info,f)


def load_data_info():

    try:
        with open(DATA_INFO_PATH,"rb") as f:
            return pickle.load(f)

    except:
        return None


# ------------------ Model Training ------------------ #

def create_and_train_model(data_dir, epochs=2, existing_model=None):

    images, labels, class_names = load_dataset(data_dir)

    if images.size == 0:
        return class_names, None

    num_classes = len(class_names)

    if existing_model:

        model = existing_model
        print("Continuing training existing model")

    else:

        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(224,224,3)
        )

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128,activation="relu")(x)

        predictions = tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )(x)

        model = Model(inputs=base_model.input,outputs=predictions)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy",sensitivity,specificity]
        )

    print("Training model...")

    model.fit(
        images,
        labels,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2
    )

    preds = model.predict(images)

    pred_classes = np.argmax(preds,axis=1)
    true_classes = np.argmax(labels,axis=1)

    cm = confusion_matrix(true_classes,pred_classes)

    print("Confusion Matrix")
    print(cm)

    model.save(MODEL_PATH)

    save_data_info(data_dir,class_names)

    return class_names, model


# ------------------ Image Prediction ------------------ #

def cancer_cell_chatbot(image_path,class_names):

    try:

        model = load_model(
            MODEL_PATH,
            custom_objects={
                "sensitivity":sensitivity,
                "specificity":specificity
            }
        )

        img_array = load_and_preprocess_image(image_path)

        if img_array is None:
            return "Image processing failed"

        prediction = model.predict(img_array)

        idx = np.argmax(prediction)

        class_name = class_names[idx]

        confidence = prediction[0][idx] * 100

        return f"Prediction: {class_name} ({confidence:.2f}%)"

    except Exception as e:
        return f"Error: {e}"


# ------------------ Program ------------------ #

if __name__ == "__main__":

    data_dir = input("Enter dataset path: ")

    class_names = []
    model = None

    data_info = load_data_info()

    if data_info and os.path.exists(MODEL_PATH):

        print("Existing model found")

        class_names = data_info["class_names"]

        model = load_model(
            MODEL_PATH,
            custom_objects={
                "sensitivity":sensitivity,
                "specificity":specificity
            }
        )

    else:

        print("Training new model")

        class_names, model = create_and_train_model(data_dir)

    while True:

        img_path = input(
            "\nEnter image path (or 'exit'): "
        )

        if img_path.lower() == "exit":
            break

        result = cancer_cell_chatbot(img_path,class_names)

        print(result)
      upload VGG16 breast cancer detection code
