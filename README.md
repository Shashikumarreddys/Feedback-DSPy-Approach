# HelpSteer2 Prompt Optimization using DSPy MIPROv2

**Technical Project Documentation** | Audience: Senior Engineers | Status: Internal Review | Date: February 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [How DSPy Addresses the Problem](#2-how-dspy-addresses-the-problem)
3. [System Architecture](#3-system-architecture)
4. [Dataset Experiments and Selection](#4-dataset-experiments-and-selection)
5. [Evaluation Metric Design](#5-evaluation-metric-design)
6. [DSPy Module Selection](#6-dspy-module-selection)
7. [Evaluator Architecture � LLM-as-Judge](#7-evaluator-architecture--llm-as-judge-using-dspy)
8. [Model Configurations Evaluated](#8-model-configurations-evaluated)
9. [Project Structure](#9-project-structure)
10. [Installation & Setup](#installation--setup)
11. [Running the Optimizer](#running-the-optimizer)
12. [MIPROv2 Internal Optimization Flow](#12-miprov2-internal-optimization-flow)
13. [Prompt Evolution Across Optimization Runs](#13-prompt-evolution-across-optimization-runs)
14. [HelpSteer2 Attributes](#helpsteer2-attributes)
15. [References](#references)
16. [Troubleshooting](#troubleshooting)

---

## 1. Problem Statement

Response quality was initially controlled through manual prompt adjustments. When outputs appeared overly verbose, excessively technical, or structurally inconsistent, the instruction text was edited and re-tested on a small number of examples.

This approach had critical structural limitations:

| Limitation | Description |
|---|---|
| No objective evaluation | Quality judgments were subjective and non-repeatable |
| No reproducibility | Prompt revisions were tested against varying examples each time |
| Limited domain coverage | Improvements in one category degraded another |
| No systematic exploration | Only one prompt variant evaluated at a time |

A measurable, reproducible optimization framework was required.

---

## 2. How DSPy Addresses the Problem

DSPy (Declarative Self-improving Python) treats prompt instructions as tunable program parameters rather than static text. Each limitation from Section 1 maps directly to a DSPy mechanism: [dspy.ai](https://dspy.ai)

| Manual Prompting Problem | DSPy Mechanism |
|---|---|
| Subjective quality judgment | Metric function scores every response numerically on defined criteria |
| Tested against 2�3 handpicked examples | Development set of 40 diverse examples evaluated every trial |
| One prompt variant at a time | Optimizer proposes 6 instruction candidates and evaluates all |
| No reproducibility | `seed=42` ensures consistent sampling and trial ordering |
| No domain coverage | Training set spans diverse question types from HelpSteer2 |

Instead of manually writing instruction text, three components are defined: a **Signature** (input/output fields and docstring), a **Metric** (scoring function), and a **Dataset** (train + dev split). The optimizer then proposes instruction variants, evaluates them systematically, and selects the best-performing version � converting prompt engineering into a structured, measurable optimization problem.

---

## 3. System Architecture

The system consists of five components:

- **Generator** � `dspy.Predict` module that produces responses from the current instruction
- **Evaluator** � LLM-as-Judge implemented as a `dspy.Predict` module, scores each response on 5 attributes independently
- **Composite metric** � Weighted scoring function passed to the optimizer
- **MIPROv2 optimizer** � Proposes and searches over instruction candidates via Bayesian optimization
- **Train/dev split** � 160 training examples + 40 development examples from HelpSteer2

\\\
Signature + Metric + Dataset
          
   MIPROv2.compile()
          
  Phase 1: Bootstrap demos
  (proposal context only)
          
  Phase 2: Generate 6 instruction
  variants via GroundedProposer
          
  Phase 3: Bayesian search
  (10 trials over 6 instructions)
          
  Best instruction selected
          
  optimized_program.json saved
\\\

---

## 4. Dataset Experiments and Selection

HelpSteer2 (NVIDIA) contains 21,362 prompt�response pairs, each annotated by human raters on five quality dimensions scored 0�4: [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)

| Attribute | Description |
|---|---|
| helpfulness | Degree to which the response answers the question |
| correctness | Factual accuracy |
| coherence | Logical organization and clarity |
| complexity | Required domain expertise level (2 = ideal) |
| verbosity | Appropriateness of response length (2 = ideal) |

### 4.1 High-Quality Subset Experiment

**Filter applied:** \helpfulness = 4\

**Observation:** Baseline performance was strong. Optimization produced improvement with a well-refined starting instruction. High-quality examples represent near-optimal responses, providing a high starting point for the optimizer to refine instruction boundaries.

**Outcome:** Viable for optimization. Suitable when the objective is fine-tuning an already-capable instruction rather than recovering from a low baseline.

### 4.2 Middle-Quality Subset Experiment

**Filter applied:** \helpfulness = 2�3\

**Observation:** The middle-quality range provided a lower starting baseline with measurable headroom. The optimizer demonstrated stronger improvement margins from this starting point.

**Outcome:** Viable for optimization. Suitable when the objective is demonstrating larger absolute improvement from a realistic baseline.

**Both subsets were evaluated and produced valid optimization results. The middle-quality subset was selected as the primary configuration for the main runs due to its wider improvement range and realistic domain distribution.**

**Data pipeline (middle-quality):**

\\\
21,362 examples  (HelpSteer2 full train split)
        Filter: helpfulness 2�3
  2,000 examples
        Remove adversarial / jailbreak prompts
  1,713 examples
        Random sample (seed=42)
    200 examples
        Split
  160 training  +  40 development
\\\

---

## 5. Evaluation Metric Design

### Composite Score Formula

\\\
score = (helpfulness / 4)  0.30
      + (correctness / 4)  0.25
      + (coherence   / 4)  0.25
      + complexity_map[complexity]  0.10
      + verbosity_map[verbosity]    0.10
\\\

### Normalization

**Linear** (helpfulness, correctness, coherence):
\\\
normalized = raw_score / 4
\\\

**Peaked non-linear** (complexity, verbosity):
\\\python
complexity_map = {0: 0.0,  1: 0.7,  2: 1.0,  3: 0.3,  4: 0.0}
verbosity_map  = {0: 0.0,  1: 0.3,  2: 1.0,  3: 0.3,  4: 0.0}
\\\

**Rationale:** For complexity and verbosity, score=2 is the target. Score=3 or 4 indicates the response is too complex or too long � a quality degradation, not an improvement. A linear map would incorrectly reward higher scores. The peaked map penalizes deviation in either direction from the optimal midpoint.

### Weight Distribution

| Attribute | Weight | Rationale |
|---|---|---|
| helpfulness | 0.30 | Primary quality dimension |
| correctness | 0.25 | Core factual reliability |
| coherence | 0.25 | Structural clarity |
| complexity | 0.10 | Style attribute � secondary |
| verbosity | 0.10 | Style attribute � secondary |

Core quality (helpfulness + correctness + coherence) accounts for **0.80** of the total score.

---

## 6. DSPy Module Selection

Two DSPy generation modules were evaluated:

| Module | Behavior |
|---|---|
| `dspy.Predict` | Sends instruction + input directly; returns the response field only |
| `dspy.ChainOfThought` | Inserts an intermediate reasoning step before the final answer |

For this task, the objective targets final response quality rather than exposing intermediate reasoning traces. Explicit reasoning content introduced variability in output length and structure relative to metric targets.

**Final configuration:**
\\\python
self.generate = dspy.Predict(HelpSteer2Signature)
\\\

---

## 7. Evaluator Architecture � LLM-as-Judge using DSPy

The evaluator is implemented as a DSPy module:

\\\python
self.evaluate_attr = dspy.Predict(EvaluationSignature)
\\\

Each attribute is scored **independently** in a separate LLM call. For every generated response, the evaluator receives:

- The original user prompt
- The generated response
- The attribute-specific rubric (e.g., complexity scoring criteria with exact threshold guidance)

And returns:
- An integer score (0�4)
- A brief justification sentence

**Per dev example:**
- 1 generator call
- 5 evaluator calls (one per attribute)
- **Total: 6 LLM calls per example per trial**

The evaluator is fully integrated into DSPy � not external scoring logic � which means it participates in the same program compilation and optimization pipeline.

---

## 8. Model Configurations Evaluated

| Run | Generator | Evaluator | Observed Outcome |
|---|---|---|---|
| 1 | GPT-4o-mini | GPT-4o-mini | Inconsistent rubric application by evaluator |
| 2 | GPT-4o-mini | GPT-4o | Evaluation stable; generation lacked depth on domain-heavy prompts |
| 3 (current) | GPT-4o | GPT-4o | Stable scoring and adequate domain depth across diverse question types |

**Why GPT-4o for the evaluator:** The evaluator applies five independent rubric definitions per response, each with specific multi-step scoring logic. GPT-4o demonstrated significantly more consistent rubric adherence across repeated evaluations of the same response.

**Why GPT-4o for the generator:** Domain-technical prompts (distributed systems, computer vision APIs, core banking) required sufficient depth to achieve correctness and complexity targets. Smaller models produced consistently shallow responses on these categories.

**Final configuration:** Generator = GPT-4o | Evaluator = GPT-4o

---

## 9. Project Structure

\\\
Feedback_dspy/
 .env
 .gitignore
 README.md
 requirements.txt
 __init__.py

 data/
    training_data.json            HelpSteer2 middle-quality subset (200 examples)
    optimized_program.json        Winning instruction (loaded at inference)
    optimization_results.json     Scores, delta, token usage, signatures

 optimizer/
    __init__.py
    mipro_optimizer.py            Main optimization entry point + MIPROv2 class

 src/
     __init__.py
     config.py                     Azure OpenAI LM configuration
     signatures.py                 HelpSteer2Signature + EvaluationSignature
     generator.py                  HelpSteer2Generator (dspy.Predict)
     evaluator.py                  HelpSteer2Evaluator (dspy.Predict  5)
     apply_signature.py            Load and apply optimized program at inference
\\\

---

## Installation & Setup

### Requirements

\\\
dspy-ai
python-dotenv
\\\

> **Note:** Only \dspy-ai\ and \python-dotenv\ are required for this pipeline. DSPy handles Azure OpenAI internally.

### 1. Clone the Repository

\\\ash
git clone <your-repo-url>
cd Feedback_dspy
\\\

### 2. Create a Virtual Environment

\\\ash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
\\\

### 3. Install Dependencies

\\\ash
pip install -r requirements.txt
\\\

### 4. Configure Environment Variables

Create a `.env` file in your project root with your Azure OpenAI credentials:

\\\
AZURE_OPENAI_ENDPOINT=url
AZURE_OPENAI_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=model_name
AZURE_OPENAI_API_VERSION=version_of_model
\\\

> **Security:** Never commit `.env` to version control. Add it to \.gitignore\.

---

## Running the Optimizer

To execute the MIPROv2 optimization pipeline:

\\\ash
python optimizer/mipro_optimizer.py
\\\

This will:
- Load the 200-example dataset (160 train + 40 dev) from `data/training_data.json\
- Execute MIPROv2 optimization (3 phases: bootstrap, proposal, Bayesian search)
- Generate 6 instruction candidates
- Evaluate across 10 trials
- Save optimized instruction to `data/optimized_program.json\
- Log detailed results to `data/optimization_results.json\

---

## 12. MIPROv2 Internal Optimization Flow

\optimizer.compile()\ executes three sequential phases:

\\\

  INPUTS                                                  
  Baseline program | 160 train | 40 dev                  
  num_candidates=6 | num_trials=10                        
  max_bootstrapped_demos=0                                

                          
                          

  PHASE 1 � BOOTSTRAP DEMO CONSTRUCTION                  

                                                          
  For each of 6 demo set slots:                           
    1. Iterate over training examples                     
    2. Run generator  get response                       
    3. Score with helpsteer_metric()                      
    4. Accept if score passes threshold                   
    5. Stop when 3 accepted demos are collected           
                                                          
  Internal constant:                                      
  BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3            
                                                          
  Output: 6 demo sets  3 verified examples each         
  Role: Proposal context only � discarded before Phase 3  
                                                          

                          
                          

  PHASE 2 � INSTRUCTION PROPOSAL                         

                                                          
  GroundedProposer receives:                              
     Signature (introspected)                            
     Auto-generated dataset summary                      
     3 bootstrapped demo examples (context only)         
     Randomly selected prompting tip                     
                                                          
  GPT-4o generates 5 new instruction variants             
                                                          
  Final instruction candidate list (6 total):             
    [0] Original docstring � preserved, never changed     
    [1-5] LM-generated variants                           
                                                          
  Index 0 is always preserved. The optimizer cannot       
  regress below the baseline.                             
                                                          
   demo_candidates = None  (zero-shot for Phase 3)       
                                                          

                          
                          

  PHASE 3 � BAYESIAN SEARCH (Optuna TPE, 10 trials)      

                                                          
  Search space: instruction_index  {0,1,2,3,4,5}        
  No demo dimension � zero-shot evaluation only.          
                                                          
  Why 10 trials for 6 instructions?                       
                         
  TPE does not enumerate each instruction once.           
  It explores and exploits:                               
    Early trials  sample broadly across all 6 indices   
    Later trials  re-sample high-performing indices     
    Re-testing confirms score stability                   
                                                          
  Per trial: 40 examples  6 calls = 240 LLM calls       
  Full run total: ~3,600+ LLM calls                       
                                                          

                          
                          

  OUTPUT                                                  
  data/optimized_program.json    winning instruction    
  data/optimization_results.json  scores + token usage  

\\\

---

## 13. Prompt Evolution Across Optimization Runs

Four runs were executed sequentially. Runs 1�3 used the same 200 sampled examples (seed=42). Run 4 used a different example set with GPT-4o as generator.

### Score Progression

| Run | Baseline Score | Optimized Score | Improvement | Dataset |
|---|---|---|---|---|
| Run 1 | 88.96% | 90.72% | **+1.76%** | 200 examples (seed=42) |
| Run 2 | 90.72% | 91.24% | **+0.52%** | 200 examples (seed=42) |
| Run 3 | 91.24% | 91.50% | **+0.26%** | 200 examples (seed=42) |
| Run 4 | 90.76% | 91.43% | **+0.67%** | Different sample, GPT-4o |

Improvement margin decreases across Runs 1�3 as the instruction matures on the same dataset. Run 4 on a different sample produces +0.67% because fresh examples introduce new optimization signal.

### Initial Baseline (Pre-Optimization)

\\\
You are a helpful assistant. Answer the user's question clearly and accurately.
\\\

No length guidance. No structural direction. No complexity adaptation.

### Run 1 � 88.96%  90.72% (+1.76%)

**What the optimizer added:** Complexity-adaptive length rules, plain language constraint, filler phrase prohibition.

\\\
Answer the user's question concisely and clearly, tailoring your response to the
complexity of the prompt. Short, simple questions require brief answers, while
detailed or multi-part questions should be addressed with structured and thorough
responses. Use plain language, define technical terms immediately, avoid unnecessary
filler phrases, and provide direct answers optimized for user comprehension.
\\\

### Run 2 � 90.72%  91.24% (+0.52%)

**What the optimizer added:** Explicit filler phrase examples, stronger complexity framing, role context.

\\\
You are a knowledgeable and concise assistant, skilled in tailoring responses to
match the complexity of user queries. Answer the user's question directly, clearly,
and concisely. For simple questions, provide a brief and immediate answer. For
detailed or multi-part questions, craft a structured response that thoroughly
addresses all aspects of the query. Use plain and accessible language, define any
technical terms upon first use, and avoid unnecessary filler phrases like "Great
question" or "Certainly." Focus on providing accurate and optimized answers that
enhance user understanding.
\\\

### Run 3 � 91.24%  91.50% (+0.26%)

**What the optimizer adjusted:** Prescriptive tone reinforced, structural pattern unchanged. Marginal gain reflects approach to local optimum on this dataset sample.

\\\
Answer user queries by acting as a highly adaptive and knowledgeable assistant.
Tailor responses to the complexity of each question. For simple questions, provide
direct and concise answers with minimal words. For complex or multi-part inquiries,
create structured and detailed responses that address all aspects of the question.
Ensure all technical terms are defined the moment they are introduced. Avoid filler
phrases such as "Great question" or "Certainly." Instead, focus on clarity, precision,
and user comprehension while maintaining brevity and relevance. Strive to provide
optimized and context-aware answers that enhance the user's understanding.
\\\

### Run 4 � 90.76%  91.43% (+0.67%) � Different Dataset Sample

**What the optimizer added:** Tone adaptation based on input complexity, depth-adjustment rule, input-awareness framing not present in earlier versions.

\\\
Act as an adaptive and proficient assistant to respond to the user's query or
instruction. Use your ability to understand the context and complexity of the
question to tailor your response appropriately. For straightforward questions,
generate a brief and direct response. For intricate or multi-part questions, craft
a structured and comprehensive answer that covers all aspects of the inquiry while
remaining concise. Define any technical term immediately when it is introduced.
Avoid using filler phrases or excessive elaboration � focus on clarity and precision.
Formulate answers in plain, easily understandable language, addressing the user's
specific needs and ensuring relevance and accuracy. Adjust your tone and depth of
information based on the complexity of the input to provide the most effective and
context-aware response.
\\\

### Instruction Evolution Pattern

Across all four runs, the optimizer independently converged on the same structural pattern:

\\\
[Role / capability framing]
        
[Complexity-adaptive rule]
 Simple  brief  |  Complex  structured + thorough
        
[Technical term definition at point of use]
        
[Filler phrase prohibition � with concrete examples]
        
[Clarity + precision + comprehension target]
\\\

The original baseline contained none of this structure. It was discovered automatically through iterative scoring on 40 diverse development examples � not through manual authoring.

---

## HelpSteer2 Attributes

Each response is scored on five dimensions, all on a 0-4 integer scale:

| Attribute | Description | Target Score |
|-----------|-------------|--------------|
| **Helpfulness** | How well the response addresses the user's question | 4 |
| **Correctness** | Factual accuracy and reliability of the information | 4 |
| **Coherence** | Clarity, logical structure, and smooth flow | 4 |
| **Complexity** | Language difficulty appropriate for a general audience | 2 |
| **Verbosity** | Length relative to the question, concise but complete | 2 |

---

## References

| Resource | Link |
|---|---|
| DSPy Official Documentation | [dspy.ai](https://dspy.ai) |
| DSPy GitHub Repository (Stanford NLP) | [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) |
| HelpSteer2 Dataset (NVIDIA / Hugging Face) | [huggingface.co/datasets/nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) |

---

## License

This project uses the [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) dataset, which is licensed under **CC-BY-4.0**.

---
