
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_features(image):
    """Extract RGB histograms and spatial color features."""
    image = image.convert("RGB").resize((96, 96))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    features = []

    # Color histogram for each RGB channel.
    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 1),
            density=True,
        )
        features.extend(histogram.tolist())

    # Average color in a 4 x 4 grid.
    for row in np.array_split(pixels, 4, axis=0):
        for cell in np.array_split(row, 4, axis=1):
            features.extend(cell.mean(axis=(0, 1)).tolist())

    return np.asarray(features, dtype=np.float32)


def load_training_data():
    """Load labeled images from data/<category>/."""
    features = []
    labels = []

    for category in CLASSES:
        folder = DATA_DIR / category

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        for image_path in folder.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    features.append(extract_features(image))
                    labels.append(category)

            except (OSError, ValueError) as error:
                print(f"Skipping {image_path.name}: {error}")

        print(
            f"{category}: "
            f"{labels.count(category)} images loaded"
        )

    return features, labels


def train_classifier():
    """Train a KNN model from the local dataset."""
    features, labels = load_training_data()

    if len(features) < 6:
        raise ValueError(
            "Not enough readable images. Add more images to the "
            "six category folders inside data/."
        )

    if len(set(labels)) < 2:
        raise ValueError(
            "Training requires images from at least two categories."
        )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=min(5, len(features)),
            weights="distance",
        ),
    )

    model.fit(np.stack(features), labels)

    print(f"Training images: {len(features)}")
    print(f"Classes learned: {sorted(set(labels))}")

    return model, len(features), sorted(set(labels))


def predict_image(image, model):
    """Predict a category and rank the KNN neighbor-vote scores."""
    features = extract_features(image).reshape(1, -1)

    classifier = model.named_steps["kneighborsclassifier"]
    scores = model.predict_proba(features)[0]
    classes = classifier.classes_

    order = np.argsort(scores)[::-1]

    ranking = [
        {
            "category": str(classes[index]),
            "confidence": float(scores[index]),
        }
        for index in order
    ]

    return ranking[0]["category"], ranking





from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def extract_features(image):
    """Extract RGB color histograms and spatial color features."""
    image = image.convert("RGB").resize((96, 96))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    features = []

    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 1),
            density=True,
        )
        features.extend(histogram.tolist())

    for row in np.array_split(pixels, 4, axis=0):
        for cell in np.array_split(row, 4, axis=1):
            features.extend(cell.mean(axis=(0, 1)).tolist())

    return np.asarray(features, dtype=np.float32)


def load_training_data():
    """Load labeled images from data/<category>/."""
    features = []
    labels = []

    for category in CLASSES:
        folder = DATA_DIR / category

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        for path in folder.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(path) as image:
                    features.append(extract_features(image))
                    labels.append(category)
            except (OSError, ValueError) as error:
                print(f"Skipping {path.name}: {error}")

        print(f"{category}: {labels.count(category)} images")

    return features, labels


def train_classifier():
    """Train the KNN classification baseline."""
    features, labels = load_training_data()

    if len(features) < 6:
        raise ValueError(
            "Not enough images. Add images to the category folders "
            "inside the data directory."
        )

    if len(set(labels)) < 2:
        raise ValueError(
            "Training requires images from at least two categories."
        )

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=min(5, len(features)),
            weights="distance",
        ),
    )

    model.fit(np.stack(features), labels)

    return model, len(features), sorted(set(labels))


def predict_image(image, model):
    """Predict a category and rank neighbor-vote scores."""
    features = extract_features(image).reshape(1, -1)
    classifier = model.named_steps["kneighborsclassifier"]
    scores = classifier.predict_proba(features)[0]
    classes = classifier.classes_

    order = np.argsort(scores)[::-1]

    ranking = [
        {
            "category": str(classes[index]),
            "confidence": float(scores[index]),
        }
        for index in order
    ]

    return ranking[0]["category"], ranking
