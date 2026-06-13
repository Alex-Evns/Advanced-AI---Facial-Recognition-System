# Project Explanations

# Why use dataset 'CelebA'

'CelebA' dataset is being used. This selection has been made for a variety of reasons including:

1) It is widely used in adademic research - The dataset therefore has a higher validity, commonly used for training and makes it ideal for training AI model

2) 202,599 images are in this dataset - This means there is alot of variation of photos for AI models to be trained on, leading to a lower likelyhood of False Positives (FP) and False Negatives (FN) with the opposite being for True Positives (TP) and True Negatives (TN)

3) there are around 40 different attrivute labels per picture, gives a wdie varietyu of characteristics to chose from. Allows for me to create good classifications which work well togtehr for training the AI model

# What/Why are we using these classifications?

1) Glasses - Clearly visable, good avalaibility across all pictures, a strong output (TP) if expected from this classification

2) Hats - They are distinctly visable in pictures, avaliabile in alot of the pictures, a moderate output is expected (vairety of TP anf FN etc)

3) Age - A much more subjective charatersitic, creates a useful discussion around bias nd unambiouscy, a lower performance is expected in this classification

# Class In-balance Findings
| Attribute   | Positive | Negative |
| ----------- | -------- | -------- |
| Eyeglasses  | 13,193   | 189,406  |
| Wearing_Hat | 9,818    | 192,781  |
| Young       | 156,734  | 45,865   |

Key Takeaway - Clear imbalances, it is not close to even in these characterisitics, therefore, this issue is resolved through balanced sampling and suing a random seed to select an even but random amount of those photos

Balanced subsets were created to prevent majority-class bias and ensure equal representation of positive and negative classes.


# Sampling Methodology

For each classifier, 2000 positive and 2000 negative examples were randomly sampled from CelebA using a fixed random seed.

# Split Methodology

Datasets were partitioned into training, validation and testing subsets using an 80/10/10 split.

# Model Selection Justification

MobileNetV2 was selected due to its balance between computational efficiency and classification performance, making it suitable for local development on consumer hardware.

# Transfer Learning Justification

Transfer learning was employed using ImageNet-pretrained weights to improve generalisation and reduce training time.
MobileNetV2 was used as it already understands edges, hspaes ,patterns and textures, meaning we only need to teach for the classifications we are looking for the AI model to idnetify. It means we need to provide it with even less data, it improveds the performance of the model overall and it is faster then training the model completely from scratch.

Due to network issues and restrictions, i had to mannually downalod MobileNetV2 weights (in models/pretrained/...)

#  Data Augmentation

We added:

RandomFlip
RandomRotation
RandomZoom

This is because without augmentation the model would memorise the images, with it, it would learn the features of each of the photos it is being trained on.
It was applied to improve robustness and reduce overfitting



# Evaluation Methodology

Model performance was evaluated using an unseen test dataset comprising 10% of the balanced dataset.
Metrics Selection

Accuracy
Measures overall correctness.

Precision
Measures reliability of positive predictions.

Recall
Measures ability to detect positive instances.

F1 Score
Balances precision and recall.

Confusion Matrix
Provides detailed insight into classification behaviour.










# Tech Stack

| Component         | Technology   |
| ----------------- | ------------ |
| Language          | Python       |
| ML Framework      | TensorFlow   |
| CNN Architecture  | MobileNetV2  |
| Dataset           | CelebA       |
| Evaluation        | Scikit-Learn |
| Frontend          | Streamlit    |
| Version Control   | GitHub       |
| Data Manipulation | Pandas       |
| Visualisation     | Matplotlib   |

# Tech stack Reasoning

Python was selected due to its extensive machine learning ecosystem and compatibility with modern AI frameworks.

TensorFlow was selected due to its robust support for deep learning and transfer learning workflows.

MobileNetV2 was selected due to its balance between computational efficiency and classification performance.
