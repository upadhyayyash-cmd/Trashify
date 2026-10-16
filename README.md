
# Trashify ♻️

Trashify is an experimental image-based waste classification
and recycling recommendation application built with Python
and Streamlit.

## Features

- Upload a waste image for classification
- Six waste categories: cardboard, glass, metal, paper, plastic, trash
- Image feature extraction using RGB histograms and spatial colors
- K-nearest-neighbors (KNN) classification baseline
- Interactive prediction analytics using Plotly
- General recycling recommendations
- CSV export of prediction history

## Technology Stack

- Python
- Streamlit
- NumPy
- Pillow
- scikit-learn
- Pandas
- Plotly
- KaggleHub (optional dataset download)

## Project Structure

```text
Trashify/
├── app.py
├── classifier.py
├── download_dataset.py
├── recommendation.py
├── analysis.py
├── requirements.txt
├── README.md
├── THIRD_PARTY_NOTICES.md
├── .gitignore
├── data/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── model/
```

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Dataset Preparation

Place permitted, labeled images in the corresponding folders:

- data/cardboard/
- data/glass/
- data/metal/
- data/paper/
- data/plastic/
- data/trash/

The dataset downloader currently retrieves the cardboard
object-detection dataset only. Other categories must be
provided separately.

Inspect dataset formats and annotations before using images
for classification. Follow the source dataset's license and
attribution requirements.

## Run the Application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Model Limitations

The initial model uses RGB color histograms, spatial color
features, and K-nearest neighbors. It is a baseline, not a
production-grade deep-learning model.

Neighbor-vote scores are not calibrated probabilities.
Performance should be measured on separate test images
that were not used during training.

The current training function fits the model using all
images found in the category folders. For formal evaluation,
create a separate train/validation/test workflow.

## Dataset and Attribution

No third-party dataset or pretrained model is intentionally
bundled with this repository.

See THIRD_PARTY_NOTICES.md for dataset sources, licenses,
attribution, and dependency information.

## Future Improvements

- Compare KNN with a convolutional neural network
- Add precision, recall, F1-score, and a confusion matrix
- Test generalization on unseen images
- Improve the user interface and deployment workflow
