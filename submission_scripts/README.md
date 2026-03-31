# Submission Scripts

## Overview
Contained in this directory are the submission scripts for the OpenAI and Anthropic APIs. This README document explain individual design choiced made in this script. When I originally ran these experiments I refered to the MolJSON format as 'graph', hence you may see this term appear throughout the submission scripts and results files. 

## OpenAI Submission Script

Script: `submit_openai_tasks.py`

When I originally ran the OpenAI experiments at the end of December 2025 I used a different submission script which I hacked together. I have used codex (the OpenAI agentic coding tool) to tidy this submission script so it is easier to use and will allow faster reproduction of the results in my paper. I have double checked each line and run tests to confirm this submission script runs the same as my previous script, except I hope it is now more legible. If you run this submission script and find any discrepancies, please contact me and I will troubleshoot the issue. 

### Structured outputs
All the experiments utilise OpenAI structured model outputs (https://developers.openai.com/api/docs/guides/structured-outputs). This forced the LLM to output answers following a JSON schema. The MolJSON format introduced in this work is explicitly designed to be compatable with structured output modes. In order for the alternative output formats to remain comparable, and mitigate biases resulting from this sampling mode, I also used structured output schemas for all other formats. For all formats other that MolJSON, the output schema required the model emit a single keyed JSON where the key refered to the output format and the value was required to be a string. 

These schemas took the form: 

```
{
    "type": "object",
    "properties": {
        fmt: {
            "type": "string",
            "description": (
                f"Molecule written as {fmt} ONLY. "
                "Do not ask clarifying questions. Do not write any comments."
            ),
        }
    },
    "required": [fmt],
    "additionalProperties": False,
},
```
Where the format (fmt) is the output format, and is a string from the set {'graph', 'smiles', 'iupac', 'selfies', 'inchi', 'V2000_MOLBLOCK'}. For the integer output questions the schema description specifies an integer output (written as a string). 

### Responses API
The credits for this work were provided by OpenAI. This allowed me to run all experiments using the responses API as opposed to the batch API. The batch API is substantially more cost effective and would be a better choice when running such a large benchmark. When running my experiments I had a limited time to finish everything before my credits expired, so I chose to use the responses API as opposed to the batch API. 

### Constrained generation
The constrained generation tasks were added later. The prompts for these questions, which can be found in the questions directory, specify the constraints that the generated molecule should satisy, however the prompt itself doesn't explicitly state the output format to be used. For these questions, the output format is specified by the schema that is provided. I found that the models did not need explicit prompting to use a specific output format, and that the output schema itself was enough to induce the output format. As such, the UUIDs used for these questions follow a different standard that I used for ChemIQ and the the other questions in this benchmark. This specific question set uses question IDs that describe the constraint set used, then the submission script appends the output format to the uuid so we can keep track of the combination of prompt and output formats used. 



## Anthropic Submission Script

Script: `submit_translation_tasks_anthropic.py`

The Anthropic API submission script is as identical to the OpenAI script as possible. Since we only ran a subset of translation questions with the Claude Haiku 4.5 model, the script only runs for SMILES, IUPAC, and MolJSON translations. This script should be easily modifiable if you want to run the remaining set of benchmark questions. 

### Slight modification of MolJSON schema
The Anthropic structured output did not support the minimum and maximum constraints that were used in the specification of the 'charges' and 'aromatic_n_h' fields. As such, the MolJSON schema that was used for the anthropic model was slightly modified, using an enumeration of integers rather than defining a range. 
