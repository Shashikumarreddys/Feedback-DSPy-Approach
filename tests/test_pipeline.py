"""
End-to-End Pipeline Test
Tests generator + evaluator on random dataset examples
"""

import os
import sys
import logging
import json
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.azure_llm import configure_dspy_with_azure
from src.generator import HelpSteer2Generator
from src.evaluator import HelpSteer2Evaluator

DATA_DIR = os.path.join(project_root, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'pipeline_test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_clean_prompt(example: dict) -> bool:
    """Filter out adversarial, jailbreak, and edge-case prompts"""
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


def load_test_cases(num_samples: int = 5) -> list:
    """Load clean random test cases from training_data.json"""
    dataset_file = os.path.join(DATA_DIR, 'training_data.json')

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset with {len(dataset)} examples")

    clean = [ex for ex in dataset if is_clean_prompt(ex)]
    logger.info(f"Clean examples after filtering: {len(clean)}")

    sampled = random.sample(clean, min(num_samples, len(clean)))

    test_cases = []
    for item in sampled:
        prompt = item.get('prompt', '')
        if not prompt:
            continue
        test_cases.append({
            'prompt': prompt,
            'reference_response': item.get('response', ''),
            'reference_scores': {
                'helpfulness': item.get('helpfulness', 0),
                'correctness': item.get('correctness', 0),
                'coherence':   item.get('coherence', 0),
                'complexity':  item.get('complexity', 0),
                'verbosity':   item.get('verbosity', 0),
            },
            'reference_goodness': item.get('overall_goodness') or sum([
                item.get('helpfulness', 0),
                item.get('correctness', 0),
                item.get('coherence', 0),
                item.get('complexity', 0),
                item.get('verbosity', 0),
            ])
        })

    logger.info(f"Selected {len(test_cases)} test cases")
    for i, case in enumerate(test_cases, 1):
        logger.info(f"  {i}. {case['prompt'][:60]}...")

    return test_cases


def run_pipeline(num_samples: int = 5) -> list:
    """Run end-to-end test on random dataset examples"""
    logger.info("=" * 70)
    logger.info("END-TO-END PIPELINE TEST")
    logger.info("Generator + Evaluator on Random Dataset Examples")
    logger.info("=" * 70)

    configure_dspy_with_azure()
    generator = HelpSteer2Generator()
    evaluator = HelpSteer2Evaluator()
    test_cases = load_test_cases(num_samples)

    results = []

    for i, case in enumerate(test_cases, 1):
        prompt = case['prompt']

        logger.info(f"\n{'='*70}")
        logger.info(f"Test {i}/{len(test_cases)}")
        logger.info(f"{'='*70}")
        logger.info(f"Prompt: {prompt}")
        logger.info("-" * 70)

        try:
            logger.info("Generating response...")
            gen_result = generator(prompt=prompt)
            generated = gen_result.response
            word_count = len(generated.split())

            logger.info(f"\nGenerated Response ({word_count} words):")
            logger.info(generated[:300] + ("..." if len(generated) > 300 else ""))

            logger.info("\nEvaluating response...")
            eval_result = evaluator(prompt=prompt, response=generated)

            logger.info("\nEvaluation Scores:")
            for attr, score in eval_result.scores.items():
                ref = case['reference_scores'].get(attr, 0)
                logger.info(f"  {attr:<12}: {score}/4  (reference: {ref}/4, diff: {score - ref:+.1f})")

            goodness_diff = eval_result.goodness - case['reference_goodness']
            logger.info(f"\nOverall Goodness:   {eval_result.goodness}/20")
            logger.info(f"Reference Goodness: {case['reference_goodness']}/20")
            logger.info(f"Difference:         {goodness_diff:+.1f}")

            results.append({
                'test_number':        i,
                'prompt':             prompt,
                'generated_response': generated,
                'word_count':         word_count,
                'scores':             eval_result.scores,
                'goodness':           eval_result.goodness,
                'justifications':     eval_result.justifications,
                'reference_response': case['reference_response'],
                'reference_scores':   case['reference_scores'],
                'reference_goodness': case['reference_goodness'],
                'goodness_diff':      goodness_diff,
                'status':             'success',
            })
            logger.info("\n SUCCESS")

        except Exception as e:
            import traceback
            logger.error(f"\n FAILED: {e}")
            logger.error(traceback.format_exc())
            results.append({
                'test_number': i,
                'prompt':      prompt,
                'error':       str(e),
                'status':      'failed',
            })

    return results


def analyze(results: list) -> dict:
    """Analyze and log pipeline test results"""
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE TEST ANALYSIS")
    logger.info("=" * 70)

    successful = [r for r in results if r['status'] == 'success']
    failed     = [r for r in results if r['status'] == 'failed']

    logger.info(f"\nSuccess rate: {len(successful)}/{len(results)}")

    if not successful:
        logger.error("No successful tests to analyze")
        return {}

    avg_goodness     = sum(r['goodness']           for r in successful) / len(successful)
    avg_ref_goodness = sum(r['reference_goodness'] for r in successful) / len(successful)
    avg_words        = sum(r['word_count']          for r in successful) / len(successful)

    logger.info(f"\nOverall Statistics:")
    logger.info(f"  Generated Goodness:  {avg_goodness:.2f}/20.0 ({avg_goodness / 20 * 100:.1f}%)")
    logger.info(f"  Reference Goodness:  {avg_ref_goodness:.2f}/20.0 ({avg_ref_goodness / 20 * 100:.1f}%)")
    logger.info(f"  Average Word Count:  {avg_words:.0f} words")

    logger.info(f"\nAttribute-Level Performance:")
    logger.info(f"{'Attribute':<15} {'Generated':<12} {'Reference':<12} {'Difference'}")
    logger.info("-" * 60)

    attrs = {}
    for attr in ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity']:
        gen = sum(r['scores'][attr]           for r in successful) / len(successful)
        ref = sum(r['reference_scores'][attr] for r in successful) / len(successful)
        attrs[attr] = {'generated': gen, 'reference': ref, 'difference': gen - ref}
        logger.info(f"{attr:<15} {gen:<12.2f} {ref:<12.2f} {gen - ref:+.2f}")

    logger.info("\n" + "=" * 70)
    logger.info("QUALITY ASSESSMENT")
    logger.info("=" * 70)

    if avg_goodness >= 18.0:
        logger.info("\n EXCELLENT: Generator produces high-quality responses (18+/20)")
    elif avg_goodness >= 16.0:
        logger.info("\n GOOD: Generator meets quality targets (16-18/20)")
    elif avg_goodness >= 14.0:
        logger.info("\n ACCEPTABLE: Generator needs some improvement (14-16/20)")
    else:
        logger.info("\n POOR: Generator needs significant improvement (<14/20)")

    if 200 <= avg_words <= 250:
        logger.info(" Word count in optimal range (200-250)")
    elif 150 <= avg_words <= 300:
        logger.info(" Word count acceptable but could be tighter (150-300)")
    else:
        logger.info(" Word count outside target range")

    if failed:
        logger.warning(f"\n {len(failed)} test(s) failed")

    return {
        'success_rate':           len(successful) / len(results),
        'avg_goodness':           avg_goodness,
        'avg_reference_goodness': avg_ref_goodness,
        'avg_word_count':         avg_words,
        'attributes':             attrs,
        'num_successful':         len(successful),
        'num_failed':             len(failed),
    }


def main():
    logger.info("=" * 70)
    logger.info("END-TO-END PIPELINE TEST")
    logger.info("=" * 70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {DATA_DIR}")

    results = run_pipeline(num_samples=5)
    summary = analyze(results)

    if summary:
        results_file = os.path.join(DATA_DIR, 'pipeline_test_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'detailed_results': results},
                      f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved pipeline test results to: {results_file}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE TEST COMPLETE")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. Review detailed results in pipeline_test_results.json")
    logger.info("  2. If quality is good (16+/20), deploy to production")
    logger.info("  3. If quality needs improvement, refine signature or run more tests")


if __name__ == "__main__":
    main()
