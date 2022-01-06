import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])


def tokenize(text):
    """Tokenise text with lemmatizer and case normalisation.

    Args:
        text (str): text required to be tokenized

    Returns:
        list: tokenised list of strings
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Verb extractor using part of speech.

    Args:
        BaseEstimator (class): Base class for all estimators in scikit-learn.
        TransformerMixin (class): class for all transformers in scikit-learn.
    """

    def starting_verb(self, text):
        """Perform starting verb feature creation.

        Args:
            text (str): text that requires analysis

        Returns:
            binary: returns 1 if starting verb found or 0 if not.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        """Fit class for sklearn.

        Args:
            x (df): dataframe to perform class on
            y (df, optional): dataframe to perform class on. Defaults to None.

        Returns:
            self: class handling
        """
        return self

    def transform(self, X):
        """Transform Class for sklearn.

        Args:
            X (dataframe): dataframe that requires treatment

        Returns:
            dataframe: dataframe that has the starting verb performed on.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """Load data from disaster management db.

    Args:
        database_filepath (str): filepath to dataframe

    Returns:
        dataframe, dataframe, str: the two dataframes returned and categories
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql(database_filepath, con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    categories = Y.columns
    return X, Y, categories


def build_model():
    """Build sklearn pipeline using components.

    Returns:
        class: pipeline class
    """
    pipeline = Pipeline([
                ('features', FeatureUnion([

                    ('text_pipeline', Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer())
                    ])),

                    ('starting_verb', StartingVerbExtractor())
                ])),

                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model using sklearn classification report.

    Args:
        model (class): model built using sklearn
        X_test (dataframe): Independent variable test set
        Y_test (dataframe): Dependent variable test set
        category_names (list): list of response variable names
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    for column in category_names:
        print(column, classification_report(Y_test[column], y_pred_df[column]))


def save_model(model, model_filepath):
    """Save model to pickle file.

    Args:
        model (class): model to be saved
        model_filepath ([type]): location of model
    """
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)


def main():
    """Ochestrate code execution."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
