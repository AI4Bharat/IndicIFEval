# IndicIFEval

This repository contains the complete codebase for the IndicIFEval benchmark. 

This includes the scripts for `indicifeval-trans` creation, the pipeline for `indicifeval-ground` creation, and the custom evaluation configurations required for the `lm-evaluation-harness`.

Clone the repository and install the required dependencies.

## IndicIFEval-Trans
The `indicifeval-trans` directory contains scripts to translate the English IFEval dataset into 14 Indic languages. Navigate to this directory and execute the main translation script to generate the localized prompts.

## IndicIFEval-Ground

The `indicifeval-ground` directory houses the pipeline for synthetically generating instructions from native Indic content.

## Evaluation
We use the Language Model Evaluation Harness for benchmarking. The lm-evaluation-harness directory contains the custom configurations required for our tasks. You must run the evaluation script specifying the model and the specific task configuration.

## License
This project is licensed under the MIT License. Please refer to the LICENSE file for complete details.


