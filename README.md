# MolJSON-data

This repository contains the data and analysis scripts associated with the MolJSON paper. To use MolJSON in your own work, please see: https://github.com/oxpig/MolJSON.

> [!WARNING]
> This is the data and analysis repository for the MolJSON paper, not the main MolJSON package repository.

## Contents

- `questions/`: benchmark question files
- `model_responses/raw/`: raw model outputs
- `model_responses/checked/`: evaluated model outputs with correctness labels
- `evaluation_scripts/`: scripts used to evaluate model responses
- `analysis_scripts/`: scripts used to generate analysis figures
- `analysis_outputs/plots/`: generated figures
- `submission_scripts/`: scripts used to generate model responses

## Dependencies

To create a conda environment for the evaluation and analysis scripts:

```bash
conda create -n moljson-data -c conda-forge rdkit pandas numpy matplotlib selfies tqdm openjdk
conda activate moljson-data
```

To run the submission scripts you will also need:

```bash
pip install openai anthropic
```

## Notes

The question and response files are stored in compressed `.gz` form. These can be uncompressed with commands such as `gzip -d filename.gz`.

The `model_responses/checked` files contain the evaluated benchmark outputs and correctness labels used for analysis.

The OPSIN parser used for IUPAC-to-SMILES conversion must be downloaded separately from https://github.com/dan2097/opsin. In this work we used OPSIN version 2.9.0.

## Citation

Please use the following citation when referencing or using MolJSON.

```bibtex
@article{runcie2026moljson,
  title = {Molecular Representations for Large Language Models},
  author = {Runcie, Nicholas T. and Imrie, Fergus and Deane, Charlotte M.},
  year = {2026},
  journal = {arXiv preprint arXiv:2605.01822},
  doi = {10.48550/arXiv.2605.01822},
  url = {http://arxiv.org/abs/2605.01822},
}
```
