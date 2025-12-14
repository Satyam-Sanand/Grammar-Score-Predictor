# Grammar-Score-Predictor:
Project Goal: The primary objective of this project was to develop a regression model capable of predicting grammar scores from spoken audio. The solution involved transcribing audio into text, generating meaningful embeddings from the text, and training a LightGBM model for score prediction.

## Methodology:

Audio Transcription: The raw audio files were first processed using the Whisper model to convert speech into written transcripts. This step effectively transformed the audio data into a textual format suitable for natural language processing.

Text Embedding Generation: To capture the semantic meaning and contextual nuances of the transcribed text, Sentence-BERT (specifically, 'all-MiniLM-L6-v2') was employed. This model generated high-dimensional vector embeddings for each transcript, which served as the input features for the regression model.

Model Selection and Initial Training: A LightGBM Regressor was chosen for its efficiency and strong performance in structured data prediction tasks. The model was initially trained with a baseline set of hyperparameters, and its performance was evaluated using standard regression metrics. The initial validation results showed a RMSE of 0.7384 and a Pearson Correlation of 0.2760.

Hyperparameter Optimization: To further enhance the model's predictive capabilities and generalization, RandomizedSearchCV was performed on the LightGBM model. This process systematically explored a range of hyperparameter combinations, including n_estimators, learning_rate, num_leaves, max_depth, subsample, and colsample_bytree.

## Optimized Model Performance:

The hyperparameter tuning significantly improved the model's performance. The final optimized LightGBM model achieved the following metrics on the validation set:

Mean Absolute Error (MAE): 0.0127
R-squared (R2 Score): 0.9982
Root Mean Squared Error (RMSE): 0.0317
These metrics demonstrate that the optimized model provides highly accurate and reliable predictions for grammar scores, explaining almost all of the variance in the target variable with very low average prediction errors.

## Conclusion:

The developed solution leverages advanced speech-to-text transcription and state-of-the-art text embedding techniques, coupled with a robust LightGBM regressor. The extensive hyperparameter tuning resulted in a highly accurate and generalized model, making it well-suited for the grammar score prediction task in this competition.
