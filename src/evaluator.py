"""
HelpSteer2 Multi-Attribute Evaluator
Evaluates responses on 5 HelpSteer2 dimensions using DSPy
"""

import re
import logging
import dspy

from src.signatures import EvaluationSignature

logger = logging.getLogger(__name__)


class HelpSteer2Evaluator(dspy.Module):
    """
    Evaluates responses on 5 dimensions:
    helpfulness, correctness, coherence, complexity, verbosity.
    """

    ATTRIBUTES = {
        'helpfulness': (
            'Helpfulness: How well does the response address the user question?\n'
            'Start at 3 (good). Give 4 if exceptionally helpful with clear examples. '
            'Give 2 if partially helpful but misses key points. '
            'Give 1 if barely relevant. Give 0 if completely unhelpful.'
        ),

        'correctness': (
            'Correctness: How factually accurate is the information?\n'
            'Start at 3 (correct). Give 4 if perfectly accurate with proper nuance. '
            'Give 2 if mostly correct but with minor errors. '
            'Give 1 if significant errors. Give 0 if completely wrong.'
        ),

        'coherence': (
            'Coherence: How clear, logical, and well-structured is the response?\n'
            'Start at 3 (coherent). Give 4 if exceptionally clear with smooth flow. '
            'Give 2 if somewhat disorganized but understandable. '
            'Give 1 if confusing. Give 0 if incoherent.'
        ),

        'complexity': (
            'Complexity: How much domain expertise is required to write this response?\n'
            'Score each level as follows:\n'
            '2 = IDEAL (score: 1.0): Requires moderate knowledge, written accessibly. '
            'Technical terms explained immediately in plain language. '
            'A thoughtful non-expert follows with ease.\n'
            '1 = ACCEPTABLE (score: 0.7): Slightly simple — mostly general knowledge. '
            'Lacks some depth but still informative and useful. Not oversimplified.\n'
            '3 = SUBOPTIMAL (score: 0.3): Requires solid domain knowledge. '
            'Some jargon used without full explanation. Motivated reader can still follow.\n'
            '4 = POOR (score: 0.0): Requires deep expert knowledge. Dense academic language, '
            'unexplained technical terms, assumes PhD-level background.\n'
            '0 = POOR (score: 0.0): Incomprehensible OR insultingly basic — useless either way.\n'
            'Start at 2. Adjust UP if unexplained jargon or expert assumptions are found. '
            'Adjust DOWN only if response is trivially shallow but still useful.'
        ),

        'verbosity': (
            'Verbosity: Evaluate the amount of detail and length relative to the question.\n'
            'Score each level as follows:\n'
            '2 = IDEAL (score: 1.0): Length perfectly matches what the question needs.\n'
            '  - Simple questions (greetings, yes/no, quick facts): 20-40 words.\n'
            '  - Explanation questions (what is X, how does Y work): 80-180 words.\n'
            '  - Complex multi-part questions: 180-250 words.\n'
            '  Complete, focused, zero filler or padding.\n'
            '1 = ACCEPTABLE (score: 0.3): Slightly too brief — misses some detail '
            'but the core answer is present. User gets the gist but may need to ask more.\n'
            '3 = ACCEPTABLE (score: 0.3): Slightly too long — 250-350 words. '
            'Includes some extra context that could be trimmed but is not harmful.\n'
            '4 = POOR (score: 0.0): Exceeds 350 words. Repetitive, rambling, '
            'padded with unnecessary background or tangential information.\n'
            '0 = POOR (score: 0.0): Completely off — either a single useless word '
            'or a novel-length essay that ignores the question entirely.\n'
            'Start at 2. Estimate length relative to question type. '
            'Adjust UP if response is padded or exceeds word targets above. '
            'Adjust DOWN only if response is clearly incomplete.'
        ),
    }

    def __init__(self):
        super().__init__()
        self.evaluate_attr = dspy.Predict(EvaluationSignature)
        logger.info("Initialized HelpSteer2Evaluator with Predict")

    def forward(self, prompt: str, response: str):
        scores = {}
        justifications = {}

        for attr_name, attr_desc in self.ATTRIBUTES.items():
            result = self.evaluate_attr(
                prompt=prompt,
                response=response,
                attribute=attr_desc
            )
            scores[attr_name] = self._extract_score(result.score)
            justifications[attr_name] = getattr(result, 'justification', '')

        return dspy.Prediction(
            scores=scores,
            justifications=justifications,
            goodness=sum(scores.values()),
            prompt=prompt,
            response=response
        )

    def _extract_score(self, text: str) -> int:
        if isinstance(text, (int, float)):
            return int(max(0, min(4, text)))
        text = str(text).strip()
        match = re.search(r'\b([0-4])(?:\.\d*)?\b', text)
        if match:
            return int(match.group(1))
        logger.warning(f"Could not parse score from: '{text}', defaulting to 2")
        return 2  