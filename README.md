# Bird_Species_Recognition_AML2024
Deep learning-based classification of 200 bird species using image recognition techniques.

## Project Requirements and Setup

- ### Requirements

The project requires Python version `>=3.10,<=3.12`. All dependencies are listed in the `pyproject.toml` file.

- ### Installation

The project uses Poetry to manage dependencies. To install the dependencies, run the following command:

```bash
poetry install
```

If You don't have Poetry installed, check the [Poetry website](https://python-poetry.org/docs/).

- ### Main framework

The main framework that we use for this project is `PyTorch`.


## Possible solution

After careful consideration, the go to model seems to be either `EfficientNet` or `ResNet`. Both models have been proven to be very effective in image classification tasks. Using method such as **transfer learning** enables us to use the pre-trained weights of the model on dataset like ImageNet and fine-tune it to our specific task.

## Process

