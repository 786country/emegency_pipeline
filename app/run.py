import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.graph_objs import Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download('averaged_perceptron_tagger')
app = Flask(__name__)


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


# load data from sql db 
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)
Y = df.iloc[:, 4:]
# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    reponse_counts=Y.apply(pd.Series.value_counts)
    reponse_counts_1 =Y.apply(pd.Series.value_counts)
    reponse_counts= reponse_counts.reset_index().rename(columns={'index': 'category'})
    reponse_counts = reponse_counts.melt(id_vars=["category"], 
                 var_name="response type", 
                 value_name="Value")
    dic={}
    graphs=[]
    for cat in reponse_counts.category:

        reponse_counts_temp=reponse_counts[reponse_counts.category==cat]

        dic_2={}
        dic_2['x']=reponse_counts_temp['response type'].tolist()
        dic_2['y']=reponse_counts_temp['Value'].tolist()
        dic_2['name']=cat
        dic[cat]=dic_2
        graphs.append(dic_2)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
         {
            'data': [Bar(
            x=graphs[0]['x'],
            y=graphs[0]['y'],
            name=graphs[0]['name']), 
            Bar(
            x=graphs[1]['x'],
            y=graphs[1]['y'],
            name=graphs[1]['name']),
            Bar(
            x=graphs[2]['x'],
            y=graphs[2]['y'],
            name=graphs[2]['name'])
            ],

            'layout': {
                'title': 'Distribution of Response Value Counts',
                'barmode': 'stack',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Response Type"
                }
            }
        },
                {
            'data': [
                Heatmap(
                    x=reponse_counts_1.columns,
                    y=reponse_counts_1.index.astype(str, copy = False).tolist(),
                    z=reponse_counts_1.values.tolist()
                )
            ],

            'layout': {
                'title': 'Distribution of Response Value Counts',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Response Type"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = [f"graph-{i}" for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()