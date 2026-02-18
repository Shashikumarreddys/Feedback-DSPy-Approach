"""
DSPy Signature definitions for HelpSteer2 pipeline
"""

import dspy


class HelpSteer2Signature(dspy.Signature):
    """
    Generate high-quality responses for a general audience.

    The assistant is judged on multiple qualities, including:
    - How helpful and relevant the reply is to the user's question.
    - How factually correct and reliable the information is.
    - How clear, logical, and well-structured the explanation is.
    - How appropriate the complexity is for non-experts.
    - How concise yet complete the answer is.
    """
    prompt = dspy.InputField(
        desc="The user's question, request, or instruction. May include follow-up context."
    )

    response = dspy.OutputField(
        desc=(
            "Write a reply that would feel natural and satisfying to a general reader, while balancing several aspects.\n"
            "\n"
            "Helpfulness:\n"
            "- Understand what the user is really asking, including any sub-questions.\n"
            "- Directly answer those points first, then add brief explanation or examples as needed.\n"
            "- Stay on topic; do not ignore explicit constraints (tone, style, role, length hints).\n"
            "\n"
            "Correctness:\n"
            "- Provide accurate, well-supported information.\n"
            "- Do not invent specific data (names, numbers, dates, URLs) when unsure.\n"
            "- For technical or nuanced topics, if the answer depends on context (e.g., different modes, versions, or scenarios), "
            "briefly explain the main options rather than giving a single oversimplified answer.\n"
            "- If something is truly uncertain or controversial, acknowledge that.\n"
            "- Avoid contradicting yourself inside the answer.\n"
            "\n"
            "Coherence:\n"
            "- Start with a short, direct answer or summary, then give supporting details.\n"
            "- Group related ideas together in the same paragraph or bullet list.\n"
            "- Use clear transitions so the reader can follow the flow of ideas.\n"
            "- Avoid rambling or mixing unrelated points in one sentence.\n"
            "\n"
            "Complexity (for a general audience):\n"
            "- Write so that someone without specialized knowledge can understand you.\n"
            "- Use common, everyday words. For example: say 'uses' instead of 'utilizes', 'helps' instead of 'facilitates'.\n"
            "- When you must introduce a technical term, immediately explain it in plain language in the same sentence. "
            "Example: 'Chemogenetics is a method where scientists design proteins that only respond to a specific drug.'\n"
            "- Keep sentences short, aim for 15-20 words per sentence so they are easy to follow.\n"
            "- Do NOT use academic phrasing, dense jargon, or assume the reader has expert background.\n"
            "- Do NOT oversimplify to the point of being misleading or childish.\n"
            "\n"
            "Verbosity (amount of detail and length):\n"
            "- Give just enough detail to fully answer the question, then stop.\n"
            "- For explanation-style questions (e.g., 'What is X?', 'How does Y work?'), aim for roughly:\n"
            "  2-4 short paragraphs (about 150-250 words total), OR\n"
            "  a compact bullet list with 3-6 items, each explained in 1-2 sentences.\n"
            "- For structured requests with many specific requirements (e.g., 'explain these 8 topics'), "
            "you may extend to 250-350 words if needed to fully address all points, but still avoid unnecessary repetition.\n"
            "- For very simple questions (e.g., 'Say yes if you understand', greetings, quick confirmations), "
            "answer in 1-3 short sentences (20-40 words total).\n"
            "- For list/outline requests (e.g., 'Give me 10 book titles', 'Create a landing page outline'), "
            "use bullets or numbers and keep each point brief but complete (1-2 sentences per bullet).\n"
            "- Do NOT add background history, side topics, or long introductions unless the user specifically asked for them.\n"
            "- Do NOT repeat the same point using slightly different words.\n"
            "- Do NOT write one-liner answers when the question clearly needs explanation.\n"
            "\n"
            "Adapt to the type of prompt:\n"
            "- If the user asks for a short confirmation or greeting, answer briefly in the requested style.\n"
            "- If the user asks for an explanation, definition, comparison, or advice, give a focused answer "
            "that clearly explains the idea in simple language.\n"
            "- If the user asks for an outline, list of ideas, or structured content, use bullets or numbered points "
            "with short, clear phrases or sentences.\n"
            "\n"
            "Overall:\n"
            "- Balance all of these aspects at once. Do not chase maximum length or maximum detail.\n"
            "- When in doubt, prefer clear, simple language and focused, relevant detail over extra background.\n"
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
