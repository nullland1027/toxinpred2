import pandas as pd


def add_headers(csv_file: str, feature_num: int) -> None:
    """
    Read csv file, add headers, save to csv file.
    """
    df = pd.read_csv(csv_file, header=None)
    header = ["ID"] + ["F_" + str(i) for i in range(1, feature_num + 1)]
    df.columns = header
    df.to_csv(csv_file, index=False)


def remove_sample_id():
    pass


if __name__ == '__main__':
    filename = ["pos_main_pssm_composition.csv", "neg_main_pssm_composition.csv", "pos_alt_pssm_composition.csv",
                "neg_alt_pssm_composition.csv"]
    for i in filename:
        df = pd.read_csv(i)
        new = df.drop("ID", axis=1)
        new.to_csv(i, index=False)
