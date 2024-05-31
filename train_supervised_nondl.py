import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
reduce_stopwords = True
deletion_rate = 0.1
num_data = 50000
feature_dim = 10000
K = 5
do_weighted_sampling = False
do_random_deletion = False

# Load data
with open('dataset.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]
data = random.sample(data, num_data)  # Randomly sampling data points

with open('stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = {line.strip() for line in file}

# Preprocessing
titles = [d['title'] for d in data]
classes = [d['class'] for d in data]
if reduce_stopwords:
    titles = [" ".join([word for word in jieba.cut(title) 
                        if word not in stopwords]) for title in titles]
else:
    titles = [" ".join(jieba.cut(title)) for title in titles]

# Function for random deletion
def random_deletion(texts, deletion_rate=0.2):
    result = []
    for text in texts:
        words = text.split()
        words = [word for word in words if word not in stopwords]
        num_to_delete = max(1, int(len(words) * deletion_rate))
        for _ in range(num_to_delete):
            if words:
                words.pop(random.randint(0, len(words) - 1))
        result.append(' '.join(words))
    return result

# Mapping classes to indices
class_list = sorted(set(classes))
class_to_index = {cls: index for index, cls in enumerate(class_list)}
indices = [class_to_index[cls] for cls in classes]

# Vectorization outside the loop to get vocabulary
vectorizer = TfidfVectorizer(max_features=feature_dim,token_pattern=r'\b\w+\b')
vectorizer.fit(titles)  # Fit to obtain the feature set but don't transform here

# K-Fold cross-validation
kf = KFold(n_splits=K, shuffle=True, random_state=42)
metrics = {'precision': [], 'recall': [], 'f1': [], 'accuracy': []}


# Initialize lists to store all true labels and predictions across folds
all_true_labels = []
all_pred_labels = []

# Split data and perform cross-validation
for train_index, test_index in kf.split(titles):
    X_train_raw = [titles[i] for i in train_index]
    y_train = [indices[i] for i in train_index]
    X_test_raw = [titles[i] for i in test_index]
    y_test = [indices[i] for i in test_index]

    # Apply data augmentation on the training data only
    if do_random_deletion:
        X_train_augmented = random_deletion(X_train_raw, deletion_rate=deletion_rate)
    else:
        X_train_augmented = X_train_raw

    # Transform text data to feature vectors
    X_train = vectorizer.transform(X_train_augmented)
    X_test = vectorizer.transform(X_test_raw)

    # Weighted sampling if needed
    if do_weighted_sampling:
        sample_weights = compute_sample_weight('balanced', y_train)
        model = MultinomialNB().fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model = MultinomialNB().fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    metrics['precision'].append(precision_score(y_test, y_pred, average='macro'))
    metrics['recall'].append(recall_score(y_test, y_pred, average='macro'))
    metrics['f1'].append(f1_score(y_test, y_pred, average='macro'))
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))

    # Assuming y_test is a list of true labels for the current fold
    all_true_labels.extend(y_test)  
    # Extend the overall prediction list with the predictions for the current fold
    all_pred_labels.extend(y_pred)  

# Calculate and print average scores after cross-validation
print('Cross-Validation Metrics:')
for metric in metrics:
    average_score = np.mean(metrics[metric])
    print(f'{metric.capitalize()}: {average_score:.3f}')

# After the cross-validation loop, compute the confusion matrix for all folds
cm = confusion_matrix(all_true_labels, all_pred_labels, 
                      labels=range(len(class_list)))
class_mapping = {
    '影劇': 'Entertainment',
    '政治': 'Politics',
    '財經': 'Finance',
    '遊戲': 'Games',
    '體育': 'Sports'
}
class_list = ['影劇', '政治', '財經', '遊戲', '體育']
english_class_list = [class_mapping[cls] for cls in class_list]
# Plotting the confusion matrix
sns.set(font_scale=1.5)  # Increase the font scale factor to make fonts larger
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=english_class_list, yticklabels=english_class_list)
plt.title('Confusion Matrix across all folds')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()