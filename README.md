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

Large question and response files are stored in compressed `.gz` form.

The `model_responses/checked` files contain the evaluated benchmark outputs and correctness labels used for analysis.

Compressed files can be uncompressed with commands such as `gzip -d filename.gz`.

## Citation

Please use the following citation when referencing or using MolJSON.

```bibtex
@article{runcie2026MolJSON,
  title={},
  author={Nicholas T. Runcie and Charlotte M. Deane and Fergus Imrie},
  journal={},
  year={2026},
  doi={},
  url={},
}
```
