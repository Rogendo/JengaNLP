"""Domain-adaptive masked-language-model (MLM) pretraining.

Continue MLM training of AfriBERTa / AfroXLMR-style encoders on local text, then
use the result as ``model.base_model`` in any JengaNLP task config.
"""

from .config import CorpusSource, MLMConfig, MLMPrivacyConfig
from .data import PreparedCorpus, prepare_corpus

__all__ = ["CorpusSource", "MLMConfig", "MLMPrivacyConfig", "PreparedCorpus", "prepare_corpus"]
