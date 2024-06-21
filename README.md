# RUPBench: Benchmarking Reasoning Under Perturbations for Robustness Evaluation in Large Language Models

## Objectives
Ensuring reliable performance of large language models (LLMs) in diverse environments is crucial. Despite their successes, LLMs often struggle with adversarial inputs, impacting practical effectiveness. To evaluate LLM robustness, we introduce RUPBench, a benchmark assessing LLMs across this **15** reasoning datasets (commonsense, arithmetic, logical, knowledge-intensive) with a **nine** types of lexical, syntactic, and semantic perturbations. We conduct experiments on multiple state-of-the-art LLMs (GPT-4, Llama3, Phi-3, Gemma) on original and perturbed datasets, identifying robustness levels and common error types. Our findings highlight areas for LLM improvement to better handle diverse, noisy inputs.

## Data Construction Pipeline
The data construction pipeline is illustrated in the following figure. We start with 15 source reasoning datasets covering commonsense, logic, arithmetic, and cross-domain areas. We then apply nine general text-based perturbations at lexical, syntactic, and semantic levels, producing a total of **365,580** perturbed samples. Human experts are involved to ensure the quality and validity of these perturbations.

<div align="center">
    <img width="90%" alt="image" src="https://github.com/EternityYW/RUPBench/blob/main/image_sources/RUPBench_pipeline.png">
</div>

## Datasets
We consider 15 representative text-based source reasoning datasets. The following table provides an overview of the reasoning datasets and tasks.

<div align="center">
    <img width="90%" alt="image" src="https://github.com/EternityYW/RUPBench/blob/main/image_sources/RUPBench_data_summary.png">
</div>

For more details, please refer to the main paper Section 3.1. The source datasets (validation/test sets) are in the [./source_datasets](./source_datasets/)" folder.

## Perturbation Categories
We use the validation or test sets of each reasoning dataset as our source samples for perturbations, categorized into lexical, syntactic, and semantic types. These perturbations are designed to induce incorrect responses from the LLM while preserving the original content's essence, ensuring the ground truth answer remains unchanged.

- Lexical perturbations modify individual words to test the model’s robustness to variations. We consider three types: homophones, typos, and leetspeak.
- Syntactic perturbations alter sentence structure to evaluate the model’s understanding of grammar and sentence construction. We consider three types: It-cleft, Wh-cleft, and compound variations.
- Semantic perturbations change the meaning or context of the text to evaluate the model’s understanding of deeper linguistic aspects. We consider three types: Red herrings, CheckList (Ribeiro et al., 2020) items, and StressTest (Naik et al., 2018) statements.

The perturbed datasets are in the [./perturbed_datasets](./perturbed_datasets/)" folder. For each .csv file, the suffixes of the columns represent the following perturbations:

- Homophones: `_HF`
- Typos: `_typos`
- Leetspeak: `_Leet`
- It-cleft: `_It_cleft`
- Wh-cleft: `_Wh_cleft`
- Compound variations: `_comp`
- Red herrings: `_red_herrings`
- CheckList: `_checklist`
- StressTest: `_stress`

## Models
We evaluate several leading LLMs for RUPBench on original and perturbed samples, including GPT4o (gpt 4o, 2024), Llama3-8B-Instruct, Llama3-70B-Instruct (AI@Meta, 2024), Phi-3-mini-128kInstruct, Phi-3-medium-128k-Instruct (Abdin et al., 2024), Gemma-2B-Instruct, and Gemma-7BInstruct (Team et al., 2024). GPT-4o is accessed through the OpenAI API, while the other models are loaded from Hugging Face. For generating model responses, we use greedy decoding (temperature = 0). We demonstrate the general pipeline for running experiments on Hugging Face using sample_experiments.py.



