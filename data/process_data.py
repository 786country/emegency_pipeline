import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data for disaster management datasets and merge datasets.

    Args:
        messages_filepath (str): messages.csv filepath
        categories_filepath (str): categories.csv filepath

    Returns:
        dataframe: merged datasets
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categories = categories.categories.str.split(';',expand=True)
    row = categories.iloc[0].tolist()
    category_colnames = [i[:-2] for i in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    df = pd.concat([messages, categories], axis=1)

    return df


def clean_data(df):
    """Clean merged dataset.

    Args:
        df (dataframe): merged dataset

    Returns:
        dataframe: dataset removed of duplicates
    """
    df = df[~df.duplicated()]
    return df


def save_data(df, database_filename):
    """Save data to sqllite db.

    Args:
        df (dataframe): dataframe required for saving into db
        database_filename (str): database filename of db
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
    """Ochestrate code execution."""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        # Load data from csv
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        # Clean data
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # Save data in sqllite db
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
