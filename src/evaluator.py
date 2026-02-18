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
    All evaluations start at baseline 3 and adjust based on specific criteria.
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
            'Complexity: Evaluate the language difficulty level for a general audience.\n'
            '4 = Too complex: Dense academic language, unexplained jargon, '
            'assumes expert knowledge, very long sentences (25+ words).\n'
            '3 = Appropriately balanced: Uses everyday language, explains technical '
            'terms inline, 15-20 word sentences, accessible to non-experts.\n'
            '2 = Ideal middle ground: Simple but not childish, concrete examples, '
            'brief explanations of any technical terms.\n'
            '1 = Too simple: Oversimplified, lacks necessary detail, talks down to reader.\n'
            '0 = Extremely inappropriate: Either incomprehensibly complex or insultingly basic.\n'
            'Start at 3, then adjust based on sentence length, jargon use, and explanation quality.'
        ),

        'verbosity': (
            'Verbosity: Evaluate the amount of detail and length relative to the question.\n'
            '4 = Too verbose: Exceeds 350 words for simple questions, repetitive, '
            'includes unnecessary background, rambling.\n'
            '3 = Slightly detailed: 250-350 words, includes some extra context, '
            'could be tighter but still reasonable.\n'
            '2 = Ideal conciseness: 150-250 words for explanations, 20-40 words '
            'for simple questions, complete but focused.\n'
            '1 = Too brief: One-liner for complex questions, missing important points, incomplete.\n'
            '0 = Extremely inappropriate: Either a novel-length essay or useless one-word answer.\n'
            'Start at 3, then count words and check if length matches question complexity.'
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
        logger.warning(f"Could not parse score from: '{text}', defaulting to 3")
        return 3
