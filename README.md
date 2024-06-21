# RUPBench: Benchmarking Reasoning Under Perturbations for Robustness Evaluation in Large Language Models

## Objectives
Ensuring reliable performance of large language models (LLMs) in diverse environments is crucial. Despite their successes, LLMs often struggle with adversarial inputs, impacting practical effectiveness. To evaluate LLM robustness, we introduce RUPBench, a benchmark assessing LLMs across this **15** reasoning datasets (commonsense, arithmetic, logical, knowledge-intensive) with a **nine** types of lexical, syntactic, and semantic perturbations. We conduct experiments on multiple state-of-the-art LLMs (GPT-4, Llama3, Phi-3, Gemma) on original and perturbed datasets, identifying robustness levels and common error types. Our findings highlight areas for LLM improvement to better handle diverse, noisy inputs.

## Data Construction Pipeline

## Datasets

## Models
We evaluate several leading LLMs for RUPBench on original and perturbed samples, including GPT4o (gpt 4o, 2024), Llama3-8B-Instruct, Llama3-70B-Instruct (AI@Meta, 2024), Phi-3-mini-128kInstruct, Phi-3-medium-128k-Instruct (Abdin et al., 2024), Gemma-2B-Instruct, and Gemma-7BInstruct (Team et al., 2024). GPT-4o is accessed through the OpenAI API, while the other models are loaded from Hugging Face. For generating model responses, we use greedy decoding (temperature = 0). We demonstrate the general pipeline for running experiments on Hugging Face using sample_experiments.py.



