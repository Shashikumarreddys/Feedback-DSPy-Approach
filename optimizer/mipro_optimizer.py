"""
MIPROv2 Optimizer for HelpSteer2
Auto-optimizes HelpSteer2Signature instructions via Bayesian search
"""

import os
import sys
import json
import logging
import random
import dspy

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

from src.config import configure_dspy_with_azure
from src.generator import HelpSteer2Generator
from src.evaluator import HelpSteer2Evaluator

DATA_DIR = os.path.join(project_root, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Token Tracker ─────────────────────────────────────────────────────────────
class TokenTracker:
    def __init__(self, lm):
        self.lm = lm

    def snapshot(self):
        total_in  = sum(h.get("usage", {}).get("prompt_tokens",     0) for h in self.lm.history)
        total_out = sum(h.get("usage", {}).get("completion_tokens", 0) for h in self.lm.history)
        return total_in, total_out

    def report(self, label: str = ""):
        total_in, total_out = self.snapshot()
        input_cost  = (total_in  / 1_000_000) * 0.15
        output_cost = (total_out / 1_000_000) * 0.60
        total_cost  = input_cost + output_cost

        logger.info("─" * 50)
        logger.info(f"TOKEN & COST REPORT {label}")
        logger.info("─" * 50)
        logger.info(f"  Input  tokens : {total_in:,}")
        logger.info(f"  Output tokens : {total_out:,}")
        logger.info(f"  Total  tokens : {total_in + total_out:,}")
        logger.info(f"  Input  cost   : ${input_cost:.4f}")
        logger.info(f"  Output cost   : ${output_cost:.4f}")
        logger.info(f"  TOTAL  COST   : ${total_cost:.4f}")
        logger.info("─" * 50)

        return {
            "input_tokens":  total_in,
            "output_tokens": total_out,
            "total_tokens":  total_in + total_out,
            "input_cost":    round(input_cost,  4),
            "output_cost":   round(output_cost, 4),
            "total_cost":    round(total_cost,  4),
        }


# ── Load Dataset ──────────────────────────────────────────────────────────────
def is_clean_prompt(example: dict) -> bool:
    prompt_lower = example.get('prompt', '').lower()
    if any(w in prompt_lower for w in ['dan', 'jailbreak', 'ignore previous', 'bypass', 'do anything now']):
        return False
    if any(p in prompt_lower for p in ["let's play a game", 'quizzer', 'repeat the above']):
        return False
    if any(p in prompt_lower for p in ['kill my', 'harm ', 'hurt my']):
        return False
    if prompt_lower.count('<extra_id_1>') > 6:
        return False
    return True


def load_dataset_as_examples() -> tuple:
    dataset_file = os.path.join(DATA_DIR, 'training_data.json')

    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} records from training_data.json")

    clean = [ex for ex in dataset if is_clean_prompt(ex)]
    logger.info(f"Clean examples after filtering: {len(clean)}")

    examples = [
        dspy.Example(
            prompt      = r["prompt"],
            response    = r["response"],
            helpfulness = r.get("helpfulness", 0),
            correctness = r.get("correctness", 0),
            coherence   = r.get("coherence",   0),
            complexity  = r.get("complexity",  0),
            verbosity   = r.get("verbosity",   0),
            goodness    = r.get("goodness",    0.0),
        ).with_inputs("prompt")
        for r in clean
    ]


    sample   = random.sample(examples,200)  
    trainset = sample[:160]     # 160 train
    devset   = sample[160:]     # 40 dev  


    logger.info(f"Trainset: {len(trainset)} | Devset: {len(devset)}")
    return trainset, devset


# ── Metric — scored (0.0-1.0 float) so MIPROv2 can differentiate ─────────────
evaluator_module = None

def helpsteer_metric(example, prediction, trace=None):
    global evaluator_module
    try:
        eval_result = evaluator_module(
            prompt=example.prompt,
            response=prediction.response
        )
        s = eval_result.scores

        help_score = s["helpfulness"] / 4.0
        corr_score = s["correctness"] / 4.0
        coh_score  = s["coherence"]   / 4.0

        complexity_map = {0: 0.0, 1: 0.7, 2: 1.0, 3: 0.3, 4: 0.0}
        verbosity_map  = {0: 0.0, 1: 0.3, 2: 1.0, 3: 0.3, 4: 0.0}

        comp_score = complexity_map.get(s["complexity"], 0.0)
        verb_score = verbosity_map.get(s["verbosity"],   0.0)

        score = (
            help_score * 0.30 +
            corr_score * 0.25 +
            coh_score  * 0.25 +
            comp_score * 0.10 +
            verb_score * 0.10
        )

        return round(score, 4)

    except Exception as e:
        logger.warning(f"Metric evaluation failed: {e}")
        return 0.0

def get_signature(generate_module):
    """Works for dspy.Predict AND dspy.ChainOfThought."""
    if hasattr(generate_module, 'predict'):
        return generate_module.predict.signature   # ChainOfThought path
    return generate_module.signature               # Predict path

# ── Show What Changed in Signature ───────────────────────────────────────────
def log_signature_changes(before: dspy.Module, after: dspy.Module):
    logger.info("\n" + "=" * 70)
    logger.info("SIGNATURE CHANGES AFTER OPTIMIZATION")
    logger.info("=" * 70)

    before_sig = get_signature(before.generate)       
    after_sig  = get_signature(after.generate)

    logger.info(f"\n BEFORE:\n  '{before_sig.instructions}'")
    logger.info(f"\n AFTER :\n  '{after_sig.instructions}'")

    if before_sig.instructions != after_sig.instructions:
        logger.info("\n Signature instructions were AUTO-UPDATED by MIPROv2 ✓")
    else:
        logger.info("\n Signature instructions unchanged (baseline was already optimal)")


# ── Main Optimization ─────────────────────────────────────────────────────────
def run_optimization():
    global evaluator_module

    logger.info("=" * 70)
    logger.info("MIPROv2 OPTIMIZATION — HelpSteer2 Middle 2000")
    logger.info("=" * 70)

    lm      = configure_dspy_with_azure()
    tracker = TokenTracker(lm)

    trainset, devset = load_dataset_as_examples()

    baseline         = HelpSteer2Generator()
    evaluator_module = HelpSteer2Evaluator()

    # ── Baseline Evaluation ───────────────────────────────────────────────────
    logger.info("\nEvaluating baseline (plain signature)...")
    dspy_evaluator = Evaluate(
        devset=devset,
        metric=helpsteer_metric,
        num_threads=2,
        display_progress=True,
    )

    baseline_result = dspy_evaluator(baseline)
    baseline_score  = (float(baseline_result)  / 100.0)*100
    logger.info(f"Baseline Score: {baseline_score:.1f}%") 
    tracker.report("(after baseline)")

    # ── MIPROv2 ───────────────────────────────────────────────────────────────
    logger.info("\nStarting MIPROv2 (will auto-rewrite signature instructions)...")
    optimizer = MIPROv2(
        metric=helpsteer_metric,
        auto=None,
        num_candidates=6,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,      
        num_threads=2,
        seed=42,
    )

    optimized_program = optimizer.compile(
        baseline,
        trainset=trainset,
        valset=devset,
        num_trials=10,
        requires_permission_to_run=False,
        minibatch=False,
    )


    log_signature_changes(baseline, optimized_program)

    # ── Optimized Evaluation ──────────────────────────────────────────────────
    logger.info("\nEvaluating optimized program...")
    optimized_result = dspy_evaluator(optimized_program)
    optimized_score  = (float(optimized_result) / 100.0)*100

    logger.info("─" * 50)
    logger.info(f"Baseline Score : {baseline_score:.1f}%")
    logger.info(f"Optimized Score: {optimized_score:.1f}%")
    logger.info(f"Improvement    : {optimized_score - baseline_score:.1f}%")
    logger.info("─" * 50)

    # ── Final report — call tracker.report() ONCE ────────────────────────────
    final_tokens = tracker.report("(final)")

    optimized_program.save(os.path.join(DATA_DIR, "optimized_program.json"))
    logger.info("Saved → data/optimized_program.json")

    results = {
        "baseline_score":      round(baseline_score, 4),
        "optimized_score":     round(optimized_score, 4),
        "improvement":         round(optimized_score - baseline_score, 4),
        "original_signature":  get_signature(baseline.generate).instructions,           
        "optimized_signature": get_signature(optimized_program.generate).instructions,
        "token_usage":         final_tokens,
    }

    with open(os.path.join(DATA_DIR, "optimization_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved → data/optimization_results.json")

    return optimized_program, results


if __name__ == "__main__":
    optimized_program, results = run_optimization()  # call ONCE only

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"Score      : {results['optimized_score']:.1f}%")
    logger.info(f"Improvement: {results['improvement']:+.1f}%")
    logger.info(f"Cost       : ${results['token_usage']['total_cost']:.4f}")
    logger.info(f"\nNew Signature:\n  {results['optimized_signature']}")