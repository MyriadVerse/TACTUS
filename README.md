# TACTUS: Efficient and Effective Table-Centric Table Union Search in Data Lakes

This repository contains the official implementation of **TACTUS**, efficient and effective table-centric table union search in data lakes.

## ⚙️ Requirements

All dependencies required to run **TACTUS** can be installed with the following command:

```bash
pip install -r requirements.txt
```


## 🚀 Run the Code

This section provides instructions for running **TACTUS**.

1. Offline Process:

    ```bash
    python offline.py --data [data] --save_model
    ```

    where
    * `data`: Name of the dataset (e.g., `santosSmall`, `santosLarge`, `tusSmall`, `tusLarge`, `wiki`, `wdc`)
    * `save_model`: Optional flag to save the model

    e.g.,
    ```bash
    python offline.py --data santosSmall --save_model
    ```

2. Online Query:

    ```bash
    python query.py --data [data]
    ```

    e.g.,
    ```bash
    python query.py --data santosSmall
    ```


## 📊 Datasets

The following table summarizes the benchmarks used in our experiments and provides their download links.

| **Dataset**    | **Link**                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **SANTOS**     | [https://zenodo.org/record/7758091](https://zenodo.org/record/7758091)                                                     |
| **TUS**        | [https://github.com/RJMillerLab/table-union-search-benchmark](https://github.com/RJMillerLab/table-union-search-benchmark) |
| **Wiki Union** | [./data/wiki/Wiki-Union](./data/wiki/Wiki-Union)                                                                                                           |
| **WDC**        | [https://webdatacommons.org/webtables/](https://webdatacommons.org/webtables/)                                             |


## 🙏 Acknowledgement

We gratefully acknowledge the open-source implementation of [**Starmie**](https://github.com/megagonlabs/starmie), which provides basic components that facilitated parts of our implementation.