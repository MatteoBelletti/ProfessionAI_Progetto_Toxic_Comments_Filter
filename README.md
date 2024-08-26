# Toxic Comments Filter

## Project Overview

This project is part of the Master ProfessionAI in Data Science and focuses on developing a system to filter toxic comments. The goal is to create a machine learning model capable of identifying and filtering out comments containing toxic, abusive, or otherwise harmful language in online forums, social media platforms, or other user-generated content environments.

### Problem Description

The task involves building a text classification model that can accurately identify toxic comments from a given dataset. The model should classify comments into different categories of toxicity, such as hate speech, threats, or insults, and flag them for review or removal. This problem is framed as a Natural Language Processing (NLP) challenge, where the model must understand and interpret the nuances of human language to detect toxicity.

### Target Solution: CNN-RNN Hybrid Model with Attention

To address the challenge of toxic comment classification, a hybrid CNN-RNN model with attention mechanisms was developed. This model is optimized for multi-label text classification, enabling it to identify multiple types of toxicity within a single comment. The model architecture is designed to efficiently capture both local patterns (using CNN) and sequential dependencies (using RNN) in the text, while the attention layer allows the model to focus on the most relevant parts of the input.

Here’s a breakdown of the model architecture:

```python
def create_cnn_rnn_model(vocab_size, max_length, embedding_dim=128):
    """
    Creates a hybrid CNN-RNN model optimized with an attention layer for multi-label text classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        max_length (int): Maximum length of the input sequences.
        embedding_dim (int): Dimensionality of the embedding vector. Default is 128.

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    # Input layer
    inputs = Input(shape=(max_length,))

    # Embedding layer with regularization
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length,
                          embeddings_regularizer=l2(1e-5))(inputs)

    # CNN layers without MaxPooling
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(1e-5))(embedding)
    conv1 = BatchNormalization()(conv1)
    drop1 = Dropout(0.2)(conv1)

    conv2 = Conv1D(filters=128, kernel_size=4, activation='relu', padding='same', kernel_regularizer=l2(1e-5))(drop1)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.2)(conv2)

    # RNN layer
    gru = Bidirectional(GRU(64, return_sequences=True, kernel_regularizer=l2(1e-5), recurrent_regularizer=l2(1e-5)))(drop2)
    gru = BatchNormalization()(gru)
    drop3 = Dropout(0.2)(gru)

    # Attention layer
    attention = tf.keras.layers.Attention()([drop3, drop3])

    # Combining CNN, RNN, and Attention features
    cnn_features = GlobalMaxPooling1D()(drop2)
    rnn_features = GlobalMaxPooling1D()(drop3)
    attention_features = GlobalMaxPooling1D()(attention)
    combined = concatenate([cnn_features, rnn_features, attention_features])

    # Dense layers
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(combined)
    dense1 = BatchNormalization()(dense1)
    drop4 = Dropout(0.3)(dense1)

    # Output layer for multi-label classification
    outputs = Dense(6, activation='sigmoid')(drop4)

    # Model creation
    model = Model(inputs=inputs, outputs=outputs)

    # Model compilation with AUC-ROC metric
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            AUC(name='auc_roc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]
    )

    return model
```

#### Key Components of the Model:

- **Embedding Layer:** Converts the input words into dense vectors of fixed size. Regularization is applied to prevent overfitting.
- **CNN Layers:** Two convolutional layers capture local patterns in the text, such as phrases or short sequences of words. Batch normalization and dropout are used to improve generalization.
- **RNN Layer:** A Bidirectional GRU layer captures the sequential nature of the text, considering context from both directions (past and future).
- **Attention Layer:** This mechanism allows the model to focus on the most relevant parts of the sequence, enhancing the ability to detect toxicity even in long comments.
- **Dense Layers:** Fully connected layers refine the features extracted by the previous layers and prepare the data for multi-label classification.
- **Output Layer:** Produces a probability score for each of the six toxicity categories, using sigmoid activation to handle multi-label classification.


### Resources and References

Various resources were utilized during the project development to guide the technical and methodological choices:

- [Natural Language Toolkit (NLTK) Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

### Acknowledgments

I would like to thank the instructors and peers from the Master ProfessionAI in Data Science for their support and guidance throughout this project.

### License

This project is developed for educational purposes as part of the Master ProfessionAI program and is not intended for commercial use.
