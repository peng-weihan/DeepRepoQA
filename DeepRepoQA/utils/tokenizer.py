import os

_enc = None
_qwen_tokenizer = None
_voyageai = None


def count_tokens(content: str, model: str = "gpt-3.5-turbo") -> int:
    global _enc, _qwen_tokenizer, _voyageai

    if model.startswith("voyage") or "voyage" in model.lower():
        if _voyageai is None:
            voyageai_import_err = (
                "`voyageai` package not found, please run `pip install voyageai`"
            )
            try:
                import voyageai
            except ImportError as e:
                raise ImportError(voyageai_import_err) from e

            _voyageai = voyageai.Client()

        return _voyageai.count_tokens([content])

    # Use dedicated tokenizer for Qwen models
    if "qwen" in model.lower():
        if _qwen_tokenizer is None:
            try:
                from transformers import AutoTokenizer
                # Use the same tokenizer as the embedding model
                _qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
            except ImportError as e:
                # If transformers is not available, use character count estimation
                return len(content) // 4
            except Exception as e:
                # If loading fails, use character count estimation
                return len(content) // 4
        
        # Use Qwen dedicated tokenizer to calculate token count
        tokens = _qwen_tokenizer.encode(content, add_special_tokens=False)
        return len(tokens)

    if _enc is None:
        tiktoken_import_err = (
            "`tiktoken` package not found, please run `pip install tiktoken`"
        )
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(tiktoken_import_err) from e

        # set tokenizer cache temporarily
        should_revert = False
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            should_revert = True
            os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "_static/tiktoken_cache",
            )

        _enc = tiktoken.encoding_for_model(model)

        if should_revert:
            del os.environ["TIKTOKEN_CACHE_DIR"]

    return len(_enc.encode(content, allowed_special="all"))
