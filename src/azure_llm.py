"""
Azure OpenAI LM Configuration for DSPy
Provides a simple function to configure DSPy with Azure OpenAI
"""

import os
import logging
from dotenv import load_dotenv
import dspy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure_dspy_with_azure():
    """Configure DSPy to use Azure OpenAI via dspy.LM with azure/ prefix."""
    # Load environment variables from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)

    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_KEY')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not found in .env file")
    if not api_key:
        raise ValueError("AZURE_OPENAI_KEY not found in .env file")

    # Use the official dspy.LM with 'azure/<deployment>' format
    lm = dspy.LM(
        model=f'azure/{deployment_name}',
        api_key=api_key,
        api_base=endpoint,
        api_version=api_version,
        model_type='chat',
        temperature=0.7,
        max_tokens=800,
    )

    dspy.settings.configure(lm=lm)
    logger.info("DSPy configured successfully with Azure OpenAI")
    return lm


if __name__ == "__main__":
    logger.info("Testing Azure OpenAI connection...")
    try:
        lm = configure_dspy_with_azure()
        logger.info("SUCCESS! Azure OpenAI configured with DSPy")
    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
