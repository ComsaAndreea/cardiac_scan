from src.datasetloader import create_dataset
from src.train import train_model

if __name__ == "__main__":
    images_dir = "data/raw/imagesTr"
    masks_dir = "data/raw/labelsTr"

    # creezi dataset
    X, Y = create_dataset(images_dir, masks_dir)

    # rulezi training
    model = train_model(X, Y)