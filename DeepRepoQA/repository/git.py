import logging
import os
import shutil
from typing import Any, Dict, Optional, List

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field

from DeepRepoQA.completion.completion import CompletionModel
from DeepRepoQA.repository.file import FileRepository
from DeepRepoQA.utils.repo import maybe_clone, checkout_commit, clone_and_checkout

logger = logging.getLogger(__name__)
#TODO: Fix logic of clone and checkout,first checkout then clone then reverse original repo

class GitRepository(FileRepository):
    repo_url: Optional[str] = Field(default=None, alias="git_repo_url")
    generate_commit_message: bool = Field(default=False)
    completion: Optional[CompletionModel] = None
    current_commit: str = Field(default="")
    current_diff: Optional[str] = None
    initial_commit: str = Field(default="")

    def __init__(self, **data):
        super().__init__(**data)
        from git import Repo

        self._repo = Repo(path=self.repo_path)

        if not self._repo.heads:
            logger.error(f"Repo at {self.repo_path} has no branches")

        if data.get("commit"):
            checkout_commit(self.repo_path, data["commit"])

        self.current_commit = self._repo.head.commit.hexsha
        self.initial_commit = self.current_commit

    @classmethod
    def from_repo(cls, git_repo_url: str, repo_path: str, commit: Optional[str] = None):
        logger.info(
            f"Create GitRepository for {git_repo_url} with commit {commit} on path {repo_path} "
        )
        
        # 检查目标路径是否存在，如果存在但有问题则删除
        if os.path.exists(repo_path):
            try:
                # 尝试检查这是否是一个有效的git仓库
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=repo_path,
                    capture_output=True,
                    check=True,
                )
                # 如果是有效仓库但需要特定commit，我们检查commit是否存在
                if commit:
                    try:
                        subprocess.run(
                            ["git", "cat-file", "-e", commit],
                            cwd=repo_path,
                            capture_output=True,
                            check=True,
                        )
                        # 如果存在，直接切换到该commit
                        subprocess.run(
                            ["git", "checkout", commit],
                            cwd=repo_path,
                            capture_output=True,
                            check=True,
                        )
                        return cls(repo_path=repo_path, git_repo_url=git_repo_url, commit=commit)
                    except subprocess.CalledProcessError:
                        # commit不存在，需要重新克隆
                        logger.warning(f"Commit {commit} not found in existing repo at {repo_path}, will reclone")
                        shutil.rmtree(repo_path)
                else:
                    # 已有仓库，无需特定commit，直接使用
                    return cls(repo_path=repo_path, git_repo_url=git_repo_url, commit=commit)
            except subprocess.CalledProcessError:
                # 不是有效的git仓库，需要删除
                logger.warning(f"Directory exists at {repo_path} but is not a valid git repository, will remove and clone")
                shutil.rmtree(repo_path)
            except Exception as e:
                # 其他错误，删除目录重试
                logger.warning(f"Error checking repository at {repo_path}: {e}, will reclone")
                shutil.rmtree(repo_path)

        # 现在可以安全地克隆或检出
        if commit:
            clone_and_checkout(git_repo_url, repo_path, commit)
        else:
            maybe_clone(git_repo_url, repo_path)

        return cls(repo_path=repo_path, git_repo_url=git_repo_url, commit=commit)

    @classmethod
    def from_dict(cls, data: dict):
        return cls.from_repo(
            git_repo_url=data["repo_url"],
            repo_path=data["path"],
            commit=data["commit"],
        )

    def restore_from_snapshot(self, snapshot: dict):
        self.current_commit = snapshot["commit"]

        if snapshot.get("patch"):
            self.current_diff = snapshot["patch"]

        try:
            self.clean_untracked_files()
            self._repo.git.reset("--hard", "HEAD")  # Discard all local changes
            self._repo.git.checkout("-f", self.current_commit)  # Force checkout
        except Exception as e:
            logger.error(f"Error checking out commit {self.current_commit}: {e}")

        # TODO: Check diff and only reset changed files

    def clean_untracked_files(self):
        try:
            self._repo.git.clean("-fd")
            logger.info("Removed all untracked files.")
        except Exception as e:
            logger.error(f"Error removing untracked files: {e}")

    def dict(self):
        return {
            "type": "git",
            "repo_path": self.repo_path,
            "git_repo_url": self.repo_url,
            "commit": self.initial_commit,
        }

    def snapshot(self) -> dict:
        return {
            "commit": self.current_commit,
            "patch": self.diff(),
        }

    def create_empty_file(self, file_path: str):
        super().create_empty_file(file_path)
        self.commit(file_path)

    def save_file(self, file_path: str, updated_content: Optional[str] = None):
        file = super().save_file(file_path, updated_content)
        self.commit(file_path)
        return file

    def commit(self, file_path: str | None = None):
        commit_message = self.commit_message(file_path)

        try:
            if file_path:
                self._repo.index.add([file_path])
            else:
                self._repo.index.add("*")
            self._repo.index.commit(commit_message)
            self.current_commit = self._repo.head.commit.hexsha

            logger.info(
                f"Committed changes to git with message '{commit_message}' and commit hash '{self.current_commit}'"
            )
            self.clean_untracked_files()  # Clean untracked files after commit
        except FileNotFoundError as e:
            logger.error(
                f"Error committing changes: Current working directory not found. {e}"
            )
            # Attempt to change to the repository directory
            try:
                os.chdir(self.repo_path)
                logger.info(f"Changed working directory to {self.repo_path}")
                # Retry the commit operation
                if file_path:
                    self._repo.index.add([file_path])
                else:
                    self._repo.index.add("*")
                self._repo.index.commit(commit_message)
                self.current_commit = self._repo.head.commit.hexsha
                logger.info(
                    f"Successfully committed changes after changing directory. Commit hash: '{self.current_commit}'"
                )
            except Exception as retry_error:
                logger.error(
                    f"Failed to commit changes after changing directory: {retry_error}"
                )
        except Exception as e:
            logger.error(f"Unexpected error during commit: {e}")

    def commit_message(self, file_path: str | None = None) -> str:
        if file_path:
            diff = self._repo.git.diff("HEAD", file_path)
        else:
            diff = self._repo.git.diff("HEAD")

        if not diff:
            return "No changes."

        if self.completion and self.generate_commit_message:
            prompt = f"Generate a concise commit message for the following git diff"
            if file_path:
                prompt += f" of file {file_path}"
            prompt += f":\n\n{diff}\n\nCommit message:"

            try:
                response = self.completion.create_text_completion(
                    messages=[ChatCompletionUserMessage(role="user", content=prompt)],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error generating commit message: {e}")

        return "Automated commit by Moatless Tools"

    def diff(self, ignore_paths: Optional[List[str]] = None):
        logger.info(f"Get diff between {self.initial_commit} and {self.current_commit}")

        if ignore_paths:
            exclude_patterns = [f":(exclude){path}" for path in ignore_paths]
            diff_command = [
                self.initial_commit,
                self.current_commit,
                "--",
            ] + exclude_patterns
            return self._repo.git.diff(*diff_command)
        else:
            try:
                return self._repo.git.diff(self.initial_commit, self.current_commit)
            except Exception as e:
                logger.error(f"Error getting diff: {e}")

            if self.current_diff:
                logger.info(f"Returning cached diff")
                return self.current_diff
            else:
                return None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump.update(
            {
                "repo_path": self.repo_path,
                "git_repo_url": self.repo_url,
                "commit": self.initial_commit,
            }
        )
        return dump
