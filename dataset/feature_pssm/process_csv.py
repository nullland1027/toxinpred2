import pandas as pd


if __name__ == '__main__':
    filename = "neg_main_aac_pssm.csv"
    header = ["F_" + str(i) for i in range(1, 21)]
    df = pd.read_csv(filename, header=None)
    df.columns = header
    df.to_csv(filename, index=False)
