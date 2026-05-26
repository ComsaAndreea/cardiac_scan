# from src.LeftAtrium.datasetloader import create_dataset
# from src.LeftAtrium.train import train_model
#
# from src.LeftRightVentricle.train import train_acdc
# from src.LeftRightVentricle.test import test_patient
#
# if __name__ == "__main__":
#     images_dir = "data/raw/LeftAtrium/imagesTr"
#     masks_dir = "data/raw/LeftAtrium/labelsTr"
#
#     # creezi dataset
#     X, Y = create_dataset(images_dir, masks_dir)
#
#     # rulezi training
#     model = train_model(X, Y)
#
#     # 🔥 TRAIN
#     train_acdc("../data/raw/LRVentricle/training", epochs=3)
#
#     # 🔥 TEST
#     test_patient("../data/raw/LRVentricle/training/patient001")

from src.LeftRightVentricle_ACDC.acdc_datasetloader import load_acdc_dataset
from src.LeftRightVentricle_ACDC.unet.acdc_train import train_acdc


if __name__ == "__main__":

    root = "data/raw/LRVentricle/training"

    # 🔵 LV
    X_lv, Y_lv = load_acdc_dataset(root, target="LV")
    train_acdc(X_lv, Y_lv, "model_lv.pth", epochs=5)

    # 🔴 RV
    X_rv, Y_rv = load_acdc_dataset(root, target="RV")
    train_acdc(X_rv, Y_rv, "model_rv.pth", epochs=5)