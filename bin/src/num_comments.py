import pandas as pd
import seaborn as sns

from tqdm import tqdm
from matplotlib import pyplot as plt

from data import get_dataset, save_dataset
from reddit_api import Praw
from multiprocessing_ import parallelize_dataframe


def get_num_comments(df, reddit):
    df.loc[:, 'num_comments'] = [reddit.get_submission(id_).num_comments for id_ in tqdm(df.id.values)]
    return df


def sample_submissions(df, step: float = 0.05, values_per_step: int = 500, seed: int = 130759,
                       verbose: bool = False) -> pd.DataFrame:
    result_df = pd.DataFrame()

    for centile in [round(x / 100.0, 2) for x in range(0, 100, int(step * 100))]:
        subset = df[
            (df.num_comments >= df.num_comments.quantile(centile)) &
            (df.num_comments < df.num_comments.quantile(min(1.0, centile + step)))
            ]

        sample = subset.sample(min(values_per_step, subset.shape[0]), random_state=seed)
        result_df = result_df.append(sample)

        print(
            f'{subset.shape[0]}\t samples between ({centile}, {round(centile + step, 2)}) quantile') if verbose else None

    if verbose:
        print(f'{result_df.shape[0]} samples has been chosen.')
        plt.subplots(figsize=(24, 8))
        sns.histplot(result_df[result_df.num_comments < 200], x='num_comments', bins=200)

    return result_df


if __name__ == '__main__':
    # data = get_dataset('dataset_1.00c.csv')
    # reddit = Praw()
    #
    # num_comments_df = parallelize_dataframe(data, get_num_comments)
    # save_dataset(num_comments_df, 'num_comments.csv')

    data = get_dataset('num_comments.csv')
    sample_submissions(data, verbose=True)
