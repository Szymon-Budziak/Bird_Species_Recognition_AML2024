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


- ### Installation on Supercomputer

To setup the project on supercomputer, run the following commands:

```bash
module load Miniconda3/
eval "$(conda shell.bash hook)"

# Create new conda environment
conda create -p /path/to/venv python=3.11.10
conda activate /path/to/venv

pip install -r requirements.txt
```

- ### Main framework

The main framework that we use for this project is `PyTorch`.


## Dataset download and submit

To download the dataset, go to Kaggle webiste and in settings create Your new API token.
Then follow the instructions on [Kaggle API](https://www.kaggle.com/docs/api) how to use it in your system. Especially useful is section 
`Authentication`, so make sure to read it.

Then navigate to `data` folder in the repository and run the following command:

```bash
kaggle competitions download -c aml-2024-feather-in-focus
```

To submit a submission, run the following command:

```bash
kaggle competitions submit -c aml-2024-feather-in-focus -f submission.csv -m "Message"
```