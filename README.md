# IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in 14 Indic Languages

[![ArXiv](https://img.shields.io/badge/arXiv-2602.22125-b31b1b.svg)](https://arxiv.org/abs/2602.22125)     [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/ai4bharat/IndicIFEval) [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

IndicIFEval is an evaluation dataset to assess the instruction-following capability of LLMs in Indic languages with verifiable, rule-based constraints.

Currently it supports 14 Indic languages, in addition to English.
<table>
<tbody>
  <tr>
    <td>Assamese (as)</td>
    <td>Nepali (ne)</td>
  </tr>
  <tr>
    <td>Bengali (bn)</td>
    <td>Odia (or)</td>
  </tr>
  <tr>
    <td>Gujarati (gu)</td>
    <td>Punjabi (pa)</td>
  </tr>
  <tr>
    <td>Hindi (hi)</td>
    <td>Sanskrit (sa)</td>
  </tr>
  <tr>
    <td>Kannada (kn)</td>
    <td>Tamil (ta)</td>
  </tr>
  <tr>
    <td>Malayalam (ml)</td>
    <td>Telugu (te)</td>
  </tr>
  <tr>
    <td>Marathi (mr)</td>
    <td>Urdu (ur)</td>
  </tr>
</tbody>
</table>

This repository contains the complete codebase for the IndicIFEval benchmark from Data Creation to Benchmark Evaluation. 

This includes the scripts for `indicifeval-trans` creation, the pipeline for `indicifeval-ground` creation, and the custom evaluation configurations required for evaluation with `lm-evaluation-harness`.

## Usage

Clone the repository and install the required dependencies. Please refer to the individual README for detailed usage.

### IndicIFEval-Trans
The `indicifeval-trans` directory contains scripts to translate the English IFEval dataset into 14 Indic languages. Navigate to this directory and execute the main translation script to generate the localized prompts.

### IndicIFEval-Ground

The `indicifeval-ground` directory houses the pipeline for synthetically generating instructions from native Indic content.

### Evaluation
We use the Language Model Evaluation Harness for benchmarking. The `lm-evaluation-harness` directory contains the custom configurations required for our tasks. You must run the evaluation script specifying the model and the specific task configuration.

#### A note on `indicifeval-ground` vs. `indicifeval-trans` constraint categories

`indicifeval-ground` and `indicifeval-trans` do not use identical evaluation setups, even where a constraint category shares its name or checker code. In particular:

- **Paragraph count + first word.** `indicifeval-trans` tests `num_paragraphs` and `nth_paragraph`/`first_word` together as independent, randomly-sized constraints. In `indicifeval-ground`, this category was simplified to match how the underlying source content was collected: `num_paragraphs` is implicitly fixed at 1 and only the first-word constraint is evaluated. Released `indicifeval-ground` rows therefore omit `num_paragraphs` from `kwargs` for this instruction — this is expected, not missing data.
- **Exact-count constraints (`keywords:frequency`, `length_constraints:number_sentences`).** `indicifeval-ground` prompts for these two constraint types are always phrased as an exact count (e.g. "the word X must appear **exactly** N times"), reflecting how the source text was mined (documents where a word/sentence count matched a target exactly). The evaluation harness now honors this: when `relation` is omitted in `kwargs` for these two ground checkers, it defaults to an exact-equality comparison rather than randomly choosing "less than"/"at least". `indicifeval-trans`, by contrast, always supplies an explicit `relation` and keeps the original less-than/at-least semantics from IFEval.

#### Filtering by translation quality

Not every translated prompt released in `indicifeval-trans` was judged correct by human review. Each row carries a `tags` list that includes exactly one of `correct` or `incorrect` (translation-quality verdict), and optionally `parallel` for the subset of rows that are aligned 1:1 across all 14 languages (used for the paper's cross-lingual comparison). To reproduce paper-reported numbers, filter to rows tagged `correct` (add `parallel` as well if you need the cross-lingual-aligned subset). Use for `gold prompt` or `silver prompt` for the `indicifeval-ground` release.

## Citation

If you use IndicIFEval in your work, please cite us:

```bibtex
@article{jayakumar2026indicifeval,
      title={IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in 14 Indic Languages}, 
      author={Thanmay Jayakumar and Mohammed Safi Ur Rahman Khan and Raj Dabre and Ratish Puduppully and Anoop Kunchukuttan},
      year={2026},
      eprint={2602.22125},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.22125}, 
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Links

- [GitHub Repository 💻](https://github.com/AI4Bharat/IndicIFEval)
- [Paper 📄](https://arxiv.org/abs/2602.22125)
- [Hugging Face Dataset 🤗](https://huggingface.co/datasets/ai4bharat/IndicIFEval)


