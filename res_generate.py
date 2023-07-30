import os


def gen_res_by_t(t: float):
    dir_name = str(t)[2:]
    if not os.path.exists(f"reoccur/origin_model_res/t{dir_name}"):
        os.mkdir(f"reoccur/origin_model_res/t{dir_name}")
    os.system(
        f"python main.py -i dataset/Positive_main_dataset -t {str(t)} -o reoccur/origin_model_res/t{dir_name}/RF_pos_main.csv -m 1 -d 2")
    os.system(
        f"python main.py -i dataset/Positive_alternate_dataset -t {str(t)} -o reoccur/origin_model_res/t{dir_name}/RF_pos_alternate_realistic.csv -m 1 -d 2")
    os.system(
        f"python main.py -i dataset/Negative_main_dataset -t {str(t)} -o reoccur/origin_model_res/t{dir_name}/RF_neg_main.csv -m 1 -d 2")
    os.system(
        f"python main.py -i dataset/Negative_alternate_dataset -t {str(t)} -o reoccur/origin_model_res/t{dir_name}/RF_neg_alternate.csv -m 1 -d 2")
    os.system(
        f"python main.py -i dataset/Negative_realistic_dataset -t {str(t)} -o reoccur/origin_model_res/t{dir_name}/RF_neg_realistic.csv -m 1 -d 2")


if __name__ == '__main__':
    gen_res_by_t(0.5)
