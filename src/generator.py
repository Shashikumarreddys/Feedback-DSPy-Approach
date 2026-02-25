"""
HelpSteer2 Response Generator
Simple Predict module using HelpSteer2Signature
"""

import logging
import dspy

from src.signatures import HelpSteer2Signature

logger = logging.getLogger(__name__)


class HelpSteer2Generator(dspy.Module):
    """DSPy Predict-based generator for Azure OpenAI"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(HelpSteer2Signature)
        logger.info("Initialized HelpSteer2Generator with Predict (simple prompting)")

    def forward(self, prompt: str):
        return self.generate(prompt=prompt)