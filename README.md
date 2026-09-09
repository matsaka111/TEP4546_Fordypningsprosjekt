# TEP4541 -- Surrogate modeling for chemical kinetics

Project folder structure:

```
Fordypningsprosjekt/
├── data/
│   ├── generate_data.py    # generates training data by running Cantera simulations
│   ├── raw/                 # untouched output of generate_data.py -- never edit by hand
│   └── processed/           # train/val splits etc., derived from raw/ -- safe to regenerate
├── models/                  # saved trained model weights end up here
├── notebooks/                # exploratory work, plots, scratch analysis
├── test/                     # quick one-off test scripts (e.g. your original ignition test)
├── train.py                  # loads data/, trains h_theta(y), saves to models/
└── environment.yml           # conda environment definition
```

Workflow:
1. `conda env create -f environment.yml` then `conda activate TEP4541`
2. Run `python data/generate_data.py` to build the training dataset
   -> produces `data/raw/training_data.npz`
3. Run `python train.py` to train the surrogate model
   -> produces a saved model in `models/`

Rule of thumb: never overwrite `data/raw/` by hand -- if you need a
different version of the dataset, change `generate_data.py`'s settings
and rerun it, or add a new output filename.
