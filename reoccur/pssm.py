import numpy as np
from Bio import SeqIO
import tempfile
import numpy as pd
import os
import pickle
from Bio import SeqIO


def create_pssm(fasta_file):
    """
    读取多条序列的fasta文件，在指定文件夹下生成每一个序列的pssm文件
    :param fasta_file:
    :return:
    """
    # 读取FASTA文件
    records = SeqIO.parse(fasta_file, "fasta")

    # 遍历所有记录
    for i in range(len(records)):
        record = records[i]
        # 生成文件名，使用序列的ID
        file_name = f"{record.id}.fasta"
        path = os.path.join("..", "dataset", "feature_pssm", file_name)
        # 将序列写入文件
        SeqIO.write(record, path, "fasta")
        os.system(
            f"""psiblast -query {path} -db /Users/zhanghaohan/code/toxinpred2/Database/data \\
            -evalue 0.001 \\
            -num_iterations 3 \\
            -out_ascii_pssm /Users/zhanghaohan/code/toxinpred2/dataset/feature_pssm/out_{str(i)}.pssm""")


def extract_pssm_matrix(pssm_file):
    """
    从PSSM文件中，提取前20列纯数字矩阵
    :param pssm_file:
    :return: 2d array
    """
    matrix = []
    with open(pssm_file, 'r') as file:
        lines = file.readlines()
        # PSSM文件的前三行和最后六行是头部和尾部，我们将其忽略
        for line in lines[3:-6]:
            # 我们只关心前20列，这些列包含了20个标准氨基酸的分数
            scores = line.split()[2:22]
            # 将分数转换为整数，并添加到矩阵中
            matrix.append([score for score in scores])
    return matrix


if __name__ == '__main__':
    pass
