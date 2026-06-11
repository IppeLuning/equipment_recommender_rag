import platform
import torch
from FlagEmbedding import FlagReranker


def load_bge_reranker() -> FlagReranker:
    """
    Load BAAI/bge-reranker-v2-m3 with settings that work on both Mac and Ubuntu.
    """
    has_cuda = torch.cuda.is_available()
    is_macos = platform.system() == "Darwin"

    # fp16 is mainly useful on CUDA; avoid it on Mac/CPU
    use_fp16 = has_cuda and not is_macos

    reranker = FlagReranker(
        "BAAI/bge-reranker-v2-m3",
        use_fp16=use_fp16,
    )
    return reranker