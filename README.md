# HelpSteer2 Response Quality Pipeline

A DSPy-based pipeline for generating and evaluating high-quality responses using the NVIDIA HelpSteer2 dataset as a quality benchmark. The pipeline uses Azure OpenAI (GPT-4o) as the language model backend and evaluates responses across five quality dimensions defined by HelpSteer2.

---

## Overview

This project implements a structured prompt-engineering pipeline using [DSPy](https://dspy.ai), a framework for programming language models through declarative signatures rather than hand-written prompts. It generates responses to real user prompts sampled from the [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) dataset and evaluates them on five attributes defined in the HelpSteer2 paper.

**HelpSteer2** is a CC-BY-4.0 licensed open-source preference dataset released by NVIDIA, containing approximately 21,362 prompt-response pairs each annotated by human raters on five attributes on a 0-4 integer scale. This project filters the top 2000 high-quality examples (weighted goodness  3.5) for use in generation and evaluation testing.

The weighted goodness score used during data preparation follows the HelpSteer2 formula:

    Goodness = 0.65  helpfulness + 0.80  correctness + 0.45  coherence

**Maximum possible goodness score:** 7.6

---

## Project Structure

```
project/
 src/
    azure_llm.py              # Azure OpenAI LM configuration for DSPy
    generator.py              # HelpSteer2Generator DSPy module
    evaluator.py              # HelpSteer2Evaluator DSPy module
    signatures.py             # HelpSteer2Signature +         EvaluationSignature
 tests/
    test_pipeline.py           # End-to-end integration test
 data/
    training_data.json         # 2000 filtered high-quality 
    pipeline_test_results.json # random 5 results generated
 .env                           # API credentials (not committed to version control)
 requirements.txt               # Python dependencies
 README.md                      # This file
```

---

## HelpSteer2 Attributes

Each response is scored on five dimensions, all on a 0-4 integer scale:

| Attribute | Description | Target Score |
|-----------|-------------|--------------|
| **Helpfulness** | How well the response addresses the user's question | 4 |
| **Correctness** | Factual accuracy and reliability of the information | 4 |
| **Coherence** | Clarity, logical structure, and smooth flow | 4 |
| **Complexity** | Language difficulty appropriate for a general audience | 2-3 |
| **Verbosity** | Length relative to the question, concise but complete | 2-3 |

**Important:** For complexity and verbosity, scores of 2-3 represent the ideal range. A score of 4 means too complex or too verbose (not better). This means a "perfect" response naturally scores **4+4+4+2+2 = 16/20** or **4+4+4+3+3 = 18/20** depending on the question type.

---

## Requirements

**\
requirements.txt\**

```
dspy-ai
python-dotenv
```

> **Note:** Only \dspy-ai\ and \python-dotenv\ are required for this pipeline. DSPy handles Azure OpenAI internally.

---

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Feedback_dspy
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a \.env\ file in your project root with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=url
AZURE_OPENAI_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=model_name
AZURE_OPENAI_API_VERSION=version_of_model
```

> **Security:** Never commit \.env\ to version control. Add it to \.gitignore\.

---

## Data Preparation

The \data/training_data.json\ file contains 2000 high-quality examples sampled from the HelpSteer2 dataset. Each entry has this structure:

```json
{
  "prompt": "User question or instruction",
  "response": "Reference response from HelpSteer2",
  "helpfulness": 4,
  "correctness": 4,
  "coherence": 4,
  "complexity": 2,
  "verbosity": 2,
  "goodness": 7.0
}
```

### Data Preparation Process

The data filtering pipeline:

1. Downloads the full \
vidia/HelpSteer2\ dataset from Hugging Face (train + validation splits, ~21,362 total examples)
2. Calculates weighted goodness scores for all examples using the formula above
3. Filters examples with goodness  3.5 to retain only high-quality entries
4. Selects the top 2000 examples by goodness score
5. Saves the result to \data/training_data.json\

---

## Running the Pipeline

Execute the end-to-end pipeline test:

```bash
python tests/test_pipeline.py
```

This will:
- Load 2000 high-quality examples from \data/training_data.json\
- Generate responses using DSPy's \HelpSteer2Generator\ module
- Evaluate responses on all five HelpSteer2 attributes using \HelpSteer2Evaluator\
- Save results to \data/pipeline_test_results.json\

### Expected Output

The pipeline produces:
- **\pipeline_test_results.json\** - Generated responses with evaluation scores

---

## Architecture

### DSPy Modules

#### HelpSteer2Generator
Generates high-quality responses using DSPy's \Predict\ module with the \HelpSteer2Signature\. Takes a user prompt as input and outputs a structured response optimized for the five HelpSteer2 dimensions.

#### HelpSteer2Evaluator
Evaluates generated responses on each of the five HelpSteer2 attributes using a multi-attribute evaluation pipeline. Returns structured scores (0-4) and justifications for each dimension.

#### Azure OpenAI Integration
The pipeline uses Azure OpenAI as the language model backend, configured via the \configure_dspy_with_azure()\ function in \src/azure_llm.py\.

---

## Key Features

 **Declarative Prompting** - Uses DSPy signatures instead of hand-written prompts  
 **Multi-Attribute Evaluation** - Scores responses on all five HelpSteer2 dimensions  
 **Production-Ready** - Includes error handling, logging, and structured output  
 **Real-World Dataset** - Based on 21,000+ human-annotated examples  
 **Modular Design** - Easy to extend with new generators or evaluators  

---

## License

This project uses the [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) dataset, which is licensed under **CC-BY-4.0**.

---

## Troubleshooting

### Missing \	raining_data.json\
Ensure \data/training_data.json\ exists before running the pipeline. Regenerate from the original HelpSteer2 dataset if needed.

### Azure OpenAI Connection Issues
- Verify all environment variables in \.env\ are correctly set
- Ensure your Azure OpenAI resource has the Model deployment
- Check that your API key has the appropriate permissions

### DSPy Import Errors
Reinstall dependencies: \pip install --upgrade dspy-ai\

---

## Contributing

Contributions are welcome! Please ensure all code follows the existing structure and includes appropriate logging.
