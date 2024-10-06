# **English to Hausa / Bassa translation using Seq2Seq Model**

## **1. Dataset Creation and Preprocessing Steps**
**a. Dataset**
- The dataset used is the English-Hausa Machine Translation dataset from [Kaggle](https://www.kaggle.com/datasets/gigikenneth/englishhausa-corpus). It contains 351024 rows of parallel text, where English sentences are paired with their Hausa translations.
- The dataset used for the english to bassa translation was gotten manually from dictionaries, reason why the dataset is very small.

**b. Pre-processing**

(**Eng-Hausa**)
- Dropped the unnamed as it wasn't useful in effectively accomplishing the task.
- Renamed the source_sentence column to english and the target_sentence column to hausa.
- Dropped all rows where english or hausa had missing values as this datapoint won't really be helpful expecially if we have a sentence in english without its corresponding hausa translation and vice versa.

```python
# Droping the unnamed column
df = df.drop(columns=['Unnamed: 0'])

# Renaming the columns
df = df.rename(columns={
    'source_sentence': 'english',
    'target_sentence': 'hausa'
})

# Droping rows where 'english' or 'hausa' have missing values
df = df.dropna(subset=['english', 'hausa'])
```

- English and Hausa sentences were converted to lowercase, and special characters such as punctuation were removed using regular expressions to simplify the text.

```python
def preprocessing(text):
    # lowercase
    text = text.lower()
    # removing special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text
```

- The dataset was split into 80% training data, 10% validation data, and 10% test data.


### - **Tokenization**
- The ```Tokenizer()``` from TensorFlow was used to convert the text into sequences of integers, where each unique word was assigned an integer index. Separate tokenizers were created for both English and Hausa text.

```python
# tokenizing and converting to sequences
tokenizer_eng = Tokenizer()
tokenizer_ha = Tokenizer()

tokenizer_eng.fit_on_texts(train_df['english'])
tokenizer_ha.fit_on_texts(train_df['hausa'])

train_sequences_eng = tokenizer_eng.texts_to_sequences(train_df['english'])
train_sequences_ha = tokenizer_ha.texts_to_sequences(train_df['hausa'])

val_sequences_eng = tokenizer_eng.texts_to_sequences(val_df['english'])
val_sequences_ha = tokenizer_ha.texts_to_sequences(val_df['hausa'])

test_sequences_eng = tokenizer_eng.texts_to_sequences(test_df['english'])
test_sequences_ha = tokenizer_ha.texts_to_sequences(test_df['hausa'])
```

### - **Padding**
- ```pad_sequences()``` was used to ensure that all sequences have the same length by padding them with zeros.

```python
max_eng = max(len(seq) for seq in train_sequences_eng)
max_ha = max(len(seq) for seq in train_sequences_ha)

train_padded_eng = pad_sequences(train_sequences_eng, maxlen=max_eng, padding='post')
train_padded_ha = pad_sequences(train_sequences_ha, maxlen=max_ha, padding='post')

val_padded_eng = pad_sequences(val_sequences_eng, maxlen=max_eng, padding='post')
val_padded_ha = pad_sequences(val_sequences_ha, maxlen=max_ha, padding='post')

test_padded_eng = pad_sequences(test_sequences_eng, maxlen=max_eng, padding='post')
test_padded_ha = pad_sequences(test_sequences_ha, maxlen=max_ha, padding='post')
```

## **2. Model Architecture and design choices**
This project uses a Seq2Seq (Sequence-to-Sequence) architecture with LSTM (Long Short-Term Memory) layers for both the encoder and decoder.

1. For the encoder:
- It takes padded English sequences as input.
- Then, the embedding layer is used to convert the input sequences into dense vectors of fixed size (embedding_dim=256).
- An LSTM layer processes the embedding vectors, and it outputs two states which are passed to the decoder to help guide translation.

2. For the decoder:
- It takes the target language (Hausa) as input.
- The input is then passed through an embedding layer and an LSTM layer. The initial state of the LSTM is set to the final states of the encoder.
- The decoder produces a sequence of translated tokens, which are passed through a dense layer to generate the final word probabilities.

```python
# for the encoder
encoder_inputs = Input(shape=(max_eng,), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=vocab_size_eng, output_dim=embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
encoder_outputs, hidden_state, cell_state = encoder_lstm(encoder_embedding)
encoder_states = [hidden_state, cell_state]

# for the decoder
decoder_inputs = Input(shape=(max_ha,), name='decoder_inputs')
decoder_embedding = Embedding(input_dim=vocab_size_ha, output_dim=embedding_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_ha, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Creation of Seq2Seq model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
```


## **Design Choices:**
- The LSTM layers use 512 units to capture long-range dependencies in the input sequence.
- This architecture ensures that context from the source language (English) is transferred to the target language (Hausa) via the LSTM states.

## **3. Training process and hyperparameter**
- Epochs: 15
- Batch size: 32 (reduced for memory constraints)
- Embedding dimension: 256
- Latent dimension: 512

```python
# training the seq2seq_model
history = model.fit(
    [train_padded_eng, train_padded_ha], train_target_ha,
    epochs=15,
    batch_size=32,
    validation_data=([val_padded_eng, val_padded_ha], val_target_ha)
)
```

## **4. Insights and potential improvements**
**a. Insights**
- **Training time:** Due to the large dataset size, training time is significant even with optimizations such as reducing batch size.
- **Memory issues:** Memory constraints led to the use of reduced training data and smaller batch sizes, limiting the model’s full potential.

**b. Potential improvements**
- Further reduction of the dataset size for quicker iterations. 
- Using bidirectional LSTMs to improve context understanding.
- Implement attention mechanisms to allow the model to focus on different parts of the input sequence during translation, which would likely improve translation quality.
- Applying Byte Pair Encoding (BPE) to reduce vocabulary size and handle rare words more effectively.
- Experiment with different values for embedding dimensions, LSTM units, and batch sizes for better performance.

The **eng-bassa translation model** would have performed better if the dataset was large enough.