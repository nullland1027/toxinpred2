from Bio import SeqIO
import tempfile
import os


def write_sequences_to_temp_files(fasta_file):
    """
    从fasta文件中读取每一条序列并存储至临时文件，
    :param fasta_file:
    :return:
    """
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=True)
            with open(temp_file.name, 'w') as f:
                # 写入序列
                SeqIO.write(record, f, "fasta")

            tmp_path = temp_file.name
            seq_id = record.id
            print(f"Sequence {seq_id} written to {tmp_path}")


fasta_file = "../test_data/protein.fa"  # 你的fasta文件路径


# write_sequences_to_temp_files(fasta_file)
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
            matrix.append([int(score) for score in scores])
    return matrix


pssm_file = "../dataset/feature_pssm/test_pos_main.pssm"  # 你的PSSM文件路径
matrix = extract_pssm_matrix(pssm_file)
for i in matrix:
    print(i)
