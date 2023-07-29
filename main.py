import argparse
import warnings
import os
import re
import sys
import pandas as pd
from toxinpred2 import aac_comp, prediction, class_assignment, delete_tmp_files
from toxinpred2 import MERCI_Processor, Merci_after_processing, BLAST_processor, hybrid

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(
    description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.')

# Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Input: protein or peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output", type=str, help="Output: File for saving results by default outfile.csv")
parser.add_argument("-t", "--threshold", type=float, help="Threshold: Value between 0 to 1 by default 0.6")
parser.add_argument("-m", "--model", type=int, choices=[1, 2], help="Model: 1: AAC based RF, 2: Hybrid, by default 1")
parser.add_argument("-d", "--display", type=int, choices=[1, 2], help="Display: 1:Toxin, 2: All peptides, by default 1")
args = parser.parse_args()

if __name__ == '__main__':
    # Parameter initialization or assigning variable for command level arguments

    Sequence = args.input  # Input variable

    # Output file
    if args.output is None:
        result_filename = "outfile.csv"
    else:
        result_filename = args.output

    # Threshold
    if args.threshold is None:
        Threshold = 0.6
    else:
        Threshold = float(args.threshold)
    # Model
    if args.model is None:
        Model = int(1)
    else:
        Model = int(args.model)
    # Display
    if args.display is None:
        dplay = int(1)
    else:
        dplay = int(args.display)

    print('Summary of Parameters:')
    print('Input File: ', Sequence, '; Model: ', Model, '; Threshold: ', Threshold)
    print('Output File: ', result_filename, '; Display: ', dplay)

    # ------------------ Read input file ---------------------
    f = open(Sequence, "r")
    len1 = f.read().count('>')
    f.close()

    with open(Sequence) as f:
        records = f.read()
    records = records.split('>')[1:]
    seqid, seq = [], []

    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
        seqid.append(name)
        seq.append(sequence)
    if len(seqid) == 0:
        f = open(Sequence, "r")
        data1 = f.readlines()
        for each in data1:
            seq.append(each.replace('\n', ''))
        for i in range(1, len(seq) + 1):
            seqid.append("Seq_" + str(i))

    seqid_1 = list(map(">{}".format, seqid))
    CM = pd.concat([pd.DataFrame(seqid_1), pd.DataFrame(seq)], axis=1)
    CM.to_csv("tmp_files/Sequence_1", header=False, index=False, sep="\n")
    f.close()

    # ======================= Prediction Module start from here =====================
    if Model == 1:  # Only ML method
        aac_comp(seq, 'seq.aac')
        os.system("perl -pi -e 's/,$//g' seq.aac")  # 删除所有行末尾的逗号
        prediction('seq.aac', 'RF_model', 'seq.pred')
        class_assignment('seq.pred', Threshold, 'seq.out')
        df1 = pd.DataFrame(seqid)
        df2 = pd.DataFrame(seq)
        df3 = pd.read_csv("seq.out")
        df3 = round(df3, 3)
        df4 = pd.concat([df1, df2, df3], axis=1)
        df4.columns = ['ID', 'Sequence', 'ML_Score', 'Prediction']
        if dplay == 1:
            df4 = df4.loc[df4.Prediction == "Toxin"]
        df4.to_csv(result_filename, index=False)
        os.remove('seq.aac')
        os.remove('seq.pred')
        os.remove('seq.out')

    elif Model == 2:  # Hybrid method
        if os.path.exists('origin_docs/envfile'):
            with open('origin_docs/envfile', 'r') as file:
                data = file.readlines()
            output = []
            for line in data:
                if "#" not in line:
                    output.append(line)
            if len(output) == 4:
                paths = []
                for i in range(0, len(output)):
                    paths.append(output[i].split(':')[1].replace('\n', ''))
                blastp, blastdb, merci, motifs = paths[0], paths[1], paths[2], paths[3]
            else:
                print("Error: Please provide paths for BLAST, MERCI and required files", file=sys.stderr)
                sys.exit()
        else:
            print("Error: Provide the '{}', which comprises paths for BLAST and MERCI".format('envfile'), file=sys.stderr)
            sys.exit()
        aac_comp(input_file=seq, output_file='seq.aac')
        os.system("perl -pi -e 's/,$//g' seq.aac")  # 删除所有行末尾的逗号
        prediction('seq.aac', 'RF_model', 'seq.pred')

        # 调用BLAST
        os.system(blastp + " -task blastp -db " + blastdb + " -query " + "tmp_files/Sequence_1" + " -out tmp_files/RES_1_6_6.out -outfmt 6 -evalue 0.000001")
        os.system(merci + " -p " + "tmp_files/Sequence_1" + " -i " + motifs + " -o tmp_files/merci.txt")
        MERCI_Processor('tmp_files/merci.txt', 'tmp_files/merci_output.csv', seqid)
        Merci_after_processing('tmp_files/merci_output.csv', 'tmp_files/merci_hybrid.csv')
        BLAST_processor('tmp_files/RES_1_6_6.out', 'tmp_files/blast_hybrid.csv', seqid)
        hybrid(' tmp_files/seq.pred', seqid, 'tmp_files/merci_hybrid.csv', 'tmp_files/blast_hybrid.csv', Threshold, 'tmp_files/final_output')
        df44 = pd.read_csv('tmp_files/final_output')
        if dplay == 1:
            df44 = df44.loc[df44.Prediction == "Toxin"]
        df44 = round(df44, 3)
        df44.to_csv(result_filename, index=None)

        # delete_tmp_files()
    else:
        print("必须选择规范的模型！")
        sys.exit(1)
