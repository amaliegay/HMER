import os
from PIL import Image
import random


class DataLoader:
    def __init__(
        self,
        n_sample: int = 10,
        random_state: int = 0,
        file_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/data",
    ) -> None:
        super().__init__()
        self.n_sample = n_sample
        self.random_state = random_state
        self.file_path = file_path

        print(f"Load data from: {self.file_path}")

    def setup(self) -> None:
        with open(f"{self.file_path}/labels.txt", "r", encoding="utf-8") as labels:
            lines: list[str] = labels.readlines()
            data = {"img_names": [], "imgs": [], "expressions": []}

            random.seed(self.random_state)
            lines = random.choices(lines, k=self.n_samples)

            for line in lines:
                line_split = line.strip().split()
                img_name = line_split[0]
                expression = line_split[1:]
                with open(f"{self.file_path}/features/{img_name}.bmp", "r") as f:
                    # move image to memory immediately, avoid lazy loading,
                    # which will lead to None pointer error in loading
                    img = Image.open(f).copy()
                # Insert a tuple consist of the file name of the image, the image,
                # and the LaTeX expression in the image
                data["img_names"].append(img_name)
                data["imgs"].append(img)
                data["expressions"].append(expression)

        self.dataset = data


def build_dataset(n_sample: int, random_state: int = 0):
    dl = DataLoader(n_sample, random_state)
    dl.setup()
    data = dl.dataset
    return dl.dataset.img_names, data.imgs, data.expressions


if __name__ == "__main__":
    dataset = build_dataset
