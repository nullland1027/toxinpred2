import os
import argparse
from Bio import SeqIO
from pssmpro import pssmpro

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--inputfile", required=True)
# parser.add_argument("-o", "--outputfile", required=True)
# args = parser.parse_args()  # args即为获取的参数str形式


def create_pssm(fasta_file, out_filename):
    """
    读取多条序列的fasta文件；
    生成单独序列的fasta文件；
    在指定文件夹下生成当前序列的pssm文件；
    删除当前的fasta文件
    :param fasta_file:
    :return:
    """
    # 读取FASTA文件
    records = SeqIO.parse(fasta_file, "fasta")

    # 遍历所有记录
    for i, record in enumerate(records):
        # 生成文件名，使用序列的ID
        file_name = f"{record.id}.fasta"
        path = os.path.join("/Users", "zhanghaohan", "code", "toxinpred2", "dataset", "feature_pssm", file_name)  # fasta file path
        # 将序列写入文件
        SeqIO.write(record, path, "fasta")
        os.system(
            f"""psiblast -query {path} \\
            -db /Users/zhanghaohan/code/toxinpred2/Database/data \\
            -evalue 0.1 \\
            -num_iterations 5 \\
            -num_threads 4 \\
            -out_ascii_pssm /Users/zhanghaohan/code/toxinpred2/dataset/feature_pssm/{out_filename}_{"{:04d}".format(i + 1)}.pssm""")
        os.remove(path)


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
    # create_pssm(args.inputfile, args.outputfile)

    pssm_dir_path = "/Users/zhanghaohan/code/toxinpred2/dataset/feature_pssm/neg_main"
    feature_type = "pssm_composition"
    output_dir_path = "/Users/zhanghaohan/code/toxinpred2/dataset/feature_pssm"

    pssmpro.get_feature(pssm_dir_path, "neg_main", feature_type, output_dir_path)
