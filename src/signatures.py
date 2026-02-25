"""
DSPy Signature definitions for HelpSteer2 pipeline
"""

import dspy

class HelpSteer2Signature(dspy.Signature):
    """Act as an adaptive and proficient assistant to respond to the user's query or instruction. Use your ability to understand the context and complexity of the question to tailor your response appropriately. For straightforward questions, generate a brief and direct response. For intricate or multi-part questions, craft a structured and comprehensive answer that covers all aspects of the inquiry while remaining concise. Define any technical term immediately when it is introduced. Avoid using filler phrases or excessive elaboration—focus on clarity and precision. Formulate answers in plain, easily understandable language, addressing the user's specific needs and ensuring relevance and accuracy. Adjust your tone and depth of information based on the complexity of the input to provide the most effective and context-aware response."""

    prompt   = dspy.InputField(desc="The user's question or instruction.")
    response = dspy.OutputField(
        desc=(
            "Answer the question using only as many words as it genuinely needs. "
            "A simple question deserves a short answer. "
            "A detailed question deserves a thorough answer. "
            "A complex multi-part question deserves a structured answer — but still concise. "
            "Use plain language anyone can understand. Define any technical term the moment you use it. "
            "No bullet points unless the question asks for a list. "
            "No filler phrases like 'Great question', 'Certainly', or 'Of course'. "
            "Answer directly. Stop when the question is fully answered."
        )
    )

class EvaluationSignature(dspy.Signature):
    """Evaluate response quality on a single attribute (0-4 integer scale)"""

    prompt = dspy.InputField(desc="Original user question")
    response = dspy.InputField(desc="Generated response to evaluate")
    attribute = dspy.InputField(desc="Attribute to evaluate with detailed scoring criteria")

    score = dspy.OutputField(
        desc=(
            "Integer score from 0-4. Start evaluation assuming score is 3 (good), "
            "then adjust up to 4 if excellent or down to 2/1/0 if issues found. "
            "Return only the number."
        )
    )
    justification = dspy.OutputField(
        desc="Brief 1-2 sentence explanation for the score"
    )