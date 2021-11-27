# TwiCatch : Mature content identifier for online streaming platforms

Implementation of Graph Neural Networks to identify twitch users who use mature content during their streaming

### How to run the program

1. Clone the repo
   ```sh
   git clone https://github.com/PrasannaKumaran/TwiCatch.git
   ```
   For accounts that are SSH configured
   ```sh
    git clone git@github.com:PrasannaKumaran/TwiCatch.git
   ```
2. Install pip
   ```sh
   python -m pip install --upgrade pip
   ```
3. Create and Activate Virtual Environment (Linux)
   ```sh
   python3 -m venv [environment-name]
   source [environment-name]/bin/activate
   ```
4. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
5. Run main
   ```sh
   python3 main.py --option value
   ```
6. The following are the list of trainable parameters that can be provided in the terminal

| Option               | Description                                                                    |
| :------------------- | :----------------------------------------------------------------------------- |
| `--layer or -la`     | Graph layer name [GAT, GCN, GraphConv, GatedGraphConv, FastRGCNConv] -> string |
| `--language or -ln`  | Streamer language [DE, EN, ES, FR, PT, RU] -> string                           |
| `--hidden or -hd`    | Number of hidden layers -> int                                                 |
| `--numlayers or -nl` | Number of graph layers -> int                                                  |
| `--drop or -d`       | Drop rate -> float                                                             |
