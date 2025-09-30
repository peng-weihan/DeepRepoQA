from DeepRepoQA.codeblocks.parser.java import JavaParser
from DeepRepoQA.codeblocks.parser.parser import CodeParser
from DeepRepoQA.codeblocks.parser.python import PythonParser


def is_supported(language: str) -> bool:
    return language and language in ["python", "java"]


def create_parser_by_ext(ext: str, **kwargs) -> CodeParser | None:
    if ext == ".py":
        return PythonParser(**kwargs)
    elif ext == ".java":
        return JavaParser(**kwargs)

    raise NotImplementedError(f"Extension {ext} is not supported.")


def create_parser(language: str, **kwargs) -> CodeParser | None:
    if language == "python":
        return PythonParser(**kwargs)
    elif language == "java":
        return JavaParser(**kwargs)

    raise NotImplementedError(f"Language {language} is not supported.")
