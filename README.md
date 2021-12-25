# Mean Variance Portfolio Optimisation

## Usage
1. Create a fresh Python 3.8.10 venv.
2. Run `pip install -r requirements.txt` to install the required packages in the venv.
3. Run `pip install -e /path/to/locations/repo` in the project root to install `src` package into the venv.
4. Run `bin/main.py`, portfolio allocation will output on console and efficient frontier plot will save to root location.

## Folder Structure
├── README.md
├── bin
│   └── main.py
├── requirements.txt
├── setup.cfg
└── src
    ├── EfficientPortfolio
    │   ├── __init__.py
    │   └── portfilio.py
    └── FTX
        ├── __init__.py
        └── client.py
