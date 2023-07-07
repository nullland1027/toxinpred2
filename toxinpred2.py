import os
import sys
import numpy as np
import pandas as pd
import joblib


def aac_comp(input_file, output_file: str):
    """
    Pfeature工具可以替代，不需要该函数
    代码遍历每个氨基酸的标准缩写（即 std 列表中的元素），并统计该字符串中该氨基酸出现的次数。
    计算氨基酸数量和总长度的比例，得出每个氨基酸在该蛋白质中所占的百分比。
    :param input_file: 输入文件
    :param output_file: 输出文件
    :return:
    """
    std = list("ACDEFGHIKLMNPQRSTVWY")
    f = open(output_file, 'w')
    sys.stdout = f
    df = pd.DataFrame(input_file)
    zz = df.iloc[:, 0]
    for j in zz:
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                composition = (count / len(j)) * 100
            print("%.2f" % composition, end=",")
        print("")
    f.truncate()


def prediction(input_file, model, output_file: str):
    """
    使用加载的模型对数据进行预测，将预测结果保存为 DataFrame 中的一列，再将此列保存到输出文件中。
    :param input_file:
    :param model:
    :param output_file:
    :return:
    """
    classifier = joblib.load(model)
    X_test = np.loadtxt(input_file, delimiter=',')
    y_p_score1 = classifier.predict_proba(X_test).tolist()
    df = pd.DataFrame(y_p_score1)
    df_1 = df.iloc[:, -1]
    df_1.to_csv(output_file, index=False, header=False)


def class_assignment(input_file: str, threshold: float, output_filename: str):
    df1 = pd.read_csv(input_file, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0, len(df1)):
        if df1['ML Score'][i] >= float(threshold):
            cc.append('Toxin')
        else:
            cc.append('Non-Toxin')
    df1['Prediction'] = cc
    df1 = df1.round(3)
    df1.to_csv(output_filename, index=False)


def MERCI_Processor(merci_file, merci_processed, name):
    hh = []
    jj = []
    kk = []
    qq = []
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(merci_file) as f:
        l = []
        for line in f:
            if len(line.strip()) != 0:
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if len(hh) == 0:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('Non-Toxin')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x: x.strip(), l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq', 'Hits']] = df1.Name.str.split("(", expand=True)
        df2 = df1[['Seq', 'Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True, inplace=True)
        total_hit = int(df2.loc[len(df2) - 1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('Toxin')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Non-Toxin')
    df3 = pd.concat([pd.DataFrame(jj), pd.DataFrame(qq), pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name', 'Hits', 'Prediction']
    df3.to_csv(merci_processed, index=False)


def Merci_after_processing(merci_processed, final_merci):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name', 'Hits']]
    df5.columns = ['Subject', 'Hits']
    kk = []
    for i in range(0, len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score"] = kk
    df5 = df5[['Subject', 'MERCI Score']]
    df5.to_csv(final_merci, index=False)


def BLAST_processor(blast_result, blast_processed, name1):
    if os.stat(blast_result).st_size != 0:
        df1 = pd.read_csv(blast_result, sep="\t", header=None)
        df2 = df1.iloc[:, :2]
        df2.columns = ['Subject', 'Query']
        df3 = pd.DataFrame()
        for i in df2.Subject.unique():
            df3 = df3.append(df2.loc[df2.Subject == i][0:5]).reset_index(drop=True)
        cc = []
        for i in range(0, len(df3)):
            cc.append(df3['Query'][i].split("_")[0])
        df3['label'] = cc
        dd = []
        for i in range(0, len(df3)):
            if df3['label'][i] == 'P':
                dd.append(1)
            else:
                dd.append(-1)
        df3["vote"] = dd
        ff = []
        gg = []
        for i in df3.Subject.unique():
            ff.append(i)
            gg.append(df3.loc[df3.Subject == i]["vote"].sum())
        df4 = pd.concat([pd.DataFrame(ff), pd.DataFrame(gg)], axis=1)
        df4.columns = ['Subject', 'Blast_value']
        hh = []
        for i in range(0, len(df4)):
            if df4['Blast_value'][i] > 0:
                hh.append(0.5)
            elif df4['Blast_value'][i] == 0:
                hh.append(0)
            else:
                hh.append(-0.5)
        df4['BLAST Score'] = hh
        df4 = df4[['Subject', 'BLAST Score']]
    else:
        ss = []
        vv = []
        for j in name1:
            ss.append(j)
            vv.append(0)
        df4 = pd.concat([pd.DataFrame(ss), pd.DataFrame(vv)], axis=1)
        df4.columns = ['Subject', 'BLAST Score']
    df4.to_csv(blast_processed, index=False)


def hybrid(ml_output, name1, merci_output, blast_output, threshold, final_output):
    df6_2 = pd.read_csv(ml_output, header=None)
    df6_1 = pd.DataFrame(name1)
    df5 = pd.read_csv(merci_output)
    df4 = pd.read_csv(blast_output)
    df6 = pd.concat([df6_1, df6_2], axis=1)
    df6.columns = ['Subject', 'ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>', '')
    df7 = pd.merge(df6, df5, how='outer', on='Subject')
    df8 = pd.merge(df7, df4, how='outer', on='Subject')
    df8.fillna(0, inplace=True)
    df8['Hybrid Score'] = df8.sum(axis=1)
    df8 = df8.round(3)
    ee = []
    for i in range(0, len(df8)):
        if df8['Hybrid Score'][i] > float(threshold):
            ee.append('Toxin')
        else:
            ee.append('Non-Toxin')
    df8['Prediction'] = ee
    df8.to_csv(final_output, index=False)


def delete_tmp_files():
    os.remove('seq.aac')
    os.remove('seq.pred')
    os.remove('final_output')
    os.remove('RES_1_6_6.out')
    os.remove('merci_output.csv')
    os.remove('merci_hybrid.csv')
    os.remove('blast_hybrid.csv')
    os.remove('merci.txt')
    os.remove('Sequence_1')
