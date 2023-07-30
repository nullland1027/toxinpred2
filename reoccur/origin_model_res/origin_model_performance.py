import pandas as pd
import json
from sklearn.metrics import recall_score, roc_auc_score, matthews_corrcoef, accuracy_score, confusion_matrix


def read_files(t):
    dir_name = str(t)[2:]
    df_neg_mai = pd.read_csv(f"t{dir_name}/RF_neg_main.csv")
    df_neg_alt = pd.read_csv(f"t{dir_name}/RF_neg_alternate.csv")
    df_neg_rea = pd.read_csv(f"t{dir_name}/RF_neg_realistic.csv")
    df_pos_mai = pd.read_csv(f"t{dir_name}/RF_pos_main.csv")
    df_pos_alt_rea = pd.read_csv(f"t{dir_name}/RF_pos_alternate_realistic.csv")
    return df_neg_mai, df_neg_alt, df_neg_rea, df_pos_mai, df_pos_alt_rea


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


if __name__ == '__main__':
    df_neg_mai, df_neg_alt, df_neg_rea, df_pos_mai, df_pos_alt_rea = read_files(0.5)

    y_main = [1] * len(df_pos_mai) + [0] * len(df_neg_mai)
    y_alternate = [1] * len(df_pos_alt_rea) + [0] * len(df_neg_alt)
    y_realistic = [1] * len(df_pos_alt_rea) + [0] * len(df_neg_rea)

    pred_main, pred_alternate, pred_realistic = [], [], []

    for i in pd.concat([df_pos_mai, df_neg_mai], axis=0)["Prediction"]:
        if i == "Toxin":
            pred_main.append(1)
        elif i == "Non-Toxin":
            pred_main.append(0)

    for i in pd.concat([df_pos_alt_rea, df_neg_alt], axis=0)["Prediction"]:
        if i == "Toxin":
            pred_alternate.append(1)
        elif i == "Non-Toxin":
            pred_alternate.append(0)

    for i in pd.concat([df_pos_alt_rea, df_neg_rea], axis=0)["Prediction"]:
        if i == "Toxin":
            pred_realistic.append(1)
        elif i == "Non-Toxin":
            pred_realistic.append(0)

    performance = {
        "Main": {
            "Sens": recall_score(y_main, pred_main),
            "Spec": specificity_score(y_main, pred_main),
            "Acc":  accuracy_score(y_main, pred_main),
            "AUC":  roc_auc_score(y_main, pred_main),
            "MCC":  matthews_corrcoef(y_main, pred_main)
        },
        "Alternate": {
            "Sens": recall_score(y_alternate, pred_alternate),
            "Spec": specificity_score(y_alternate, pred_alternate),
            "Acc":  accuracy_score(y_alternate, pred_alternate),
            "AUC":  roc_auc_score(y_alternate, pred_alternate),
            "MCC":  matthews_corrcoef(y_alternate, pred_alternate)
        },
        "Realistic": {
            "Sens": recall_score(y_realistic, pred_realistic),
            "Spec": specificity_score(y_realistic, pred_realistic),
            "Acc":  accuracy_score(y_realistic, pred_realistic),
            "AUC":  roc_auc_score(y_realistic, pred_realistic),
            "MCC":  matthews_corrcoef(y_realistic, pred_realistic)
        }
    }

    with open("performance5.json", "w") as f:
        json.dump(performance, f)
    print(len(y_main), len(y_alternate), len(y_realistic))
    print(len(pred_main), len(pred_alternate), len(pred_realistic))
    print(performance)
