import data_loader
from cpm_masker import CPMMasker
from main import convert_to_wide


def main():
    dl = data_loader.DataLoader(as_r=False, protocol_c="IMAGEN.sex", file_c="mats_mid_bsl.mat", clean_data=True)
    for data_set in dl.get_data_sets():
        x = data_set.get_x()
        y = data_set.get_y()

        x = convert_to_wide(x)

        masker = CPMMasker(x, y, binary=True)


if __name__ == "__main__":
    main()
