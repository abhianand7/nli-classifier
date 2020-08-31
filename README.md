### Approach
1. Sentence Representation in Vector (1024 vector dimension)
    1. The target is to represent each sentence into a vector of same length
    2. Not cropping/padding sentences to a fixed length to avoid losing information from the sentence
    3. Tasks like NLI are sensitive to changes in the original sentence
    4. Custom Bert model used for extracting word embeddings and representing the sentence vector
        1. Each sentence is represented in a vector of fixed length of 1024.
2. Model Selection: Siamese Network
    1. Models Tried:
        1. Simple Dense Model: fits too quickly
        2. Siamese Dense Model
            1. Better at tasks which involves finding relationship between two comparable things
            2. Because of use of the sub networks and sharing of weights means fewer parameters to train for.
            3. helpful with less data and has less tendency to overfit.
        3. Siamese hybrid LSTM+CNN model
            1. Training cost is too high compared to Dense models
            2. LSTM is great for sequence learning(like in texts)
            3. Since in this case embedding is not being learnt, the performance of LSTM is comparable to that of Dense Model.
            4. So Siamese Dense Model used as the main model.

#### Results Obtained for Siamese Dense Model:
1. Accuracy Score: 79%
2. Precision: 0.8473
3. Recall: 0.7246
4. F1 Score: 0.7812
5. AUC-ROC Score: 0.89

##### steps to run the project
```
pip install -r requirements.txt
python train_bert_model.py
```

Next Steps:
1. Improve the Sentecne vector representation by fine-tuning the bert model for specifically NLI tasks.

        