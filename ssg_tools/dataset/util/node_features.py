import os
from torch_geometric.data import download_url
import pandas as pd
import numpy as np

def get_numberbatch_embeddings(classes: list, numberbatch_root: str):
    """
    Retrieves numberbatch embeddings for a given list of classes.

    Args:
        classes (list): A list of class labels.
        numberbatch_root (str): The root directory where the numberbatch file is located.

    Returns:
        pandas.DataFrame: A DataFrame containing the class embeddings.

    Raises:
        None

    Example:
        classes = ['cat', 'dog', 'bird']
        numberbatch_root = '/path/to/numberbatch'
        embeddings = get_numberbatch_embeddings(classes, numberbatch_root)
    """
    numberbatch_path = os.path.join(numberbatch_root, 'numberbatch-en-19.08.txt.gz')

    if not os.path.isfile(numberbatch_path):
        download_url('https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz', numberbatch_root)
    numberbatch = pd.read_csv(numberbatch_path, sep=" ", header=None, skiprows=1)
    numberbatch.rename(columns={0: 'word'}, inplace=True)

    class_embeddings = pd.DataFrame(data={'label': classes})
    # replace spaces with underscores for numberbatch (not necessary in rio27 but keep it for historic reasons)
    class_embeddings['_label'] = class_embeddings['label'].str.replace(" ", "_")

    # merge with numberbatch to get embeddings
    class_embeddings = pd.merge(class_embeddings, numberbatch, left_on="_label", right_on="word", how="left")

    # replace nan values with zeros for '-'
    class_embeddings.replace(np.nan, 0.0, inplace=True)

    # # get all classes without embeddings (not necessary in rio27 but keep it for historic reasons)
    # # works only on classes.txt from 3RScan
    # nan_class_embeddings = class_embeddings[class_embeddings.isna().any(axis=1)]['label']
    # # get hypernyms for classes without embeddings
    # nan_hypernyms = classes_full.loc[classes_full[1].isin(nan_class_embeddings), 3]
    # nan_hypernyms.index = nan_class_embeddings.index
    # # merge with numberbatch to get embeddings for hypernyms
    # nan_df = pd.DataFrame(data={'label': nan_class_embeddings, 'hypernym': nan_hypernyms}, index=nan_class_embeddings.index)
    # nan_df = pd.merge(nan_df, numberbatch, left_on='hypernym', right_on='word', how='left')
    # nan_df.index = nan_class_embeddings.index

    # # replace nan values with hypernym embeddings
    # class_embeddings.loc[nan_df.label.index, 'word':300] = nan_df.loc[:, 'word':300]

    return class_embeddings.drop(columns=['_label', 'word'])