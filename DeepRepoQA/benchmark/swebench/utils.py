import fcntl
import logging
import os
import shutil
from typing import Optional

from DeepRepoQA.benchmark.utils import (
    get_missing_files,
    get_missing_spans,
)
from DeepRepoQA.index import CodeIndex
from DeepRepoQA.repository import GitRepository
from DeepRepoQA.repository.repository import Repository
from DeepRepoQA.utils.repo import (
    setup_github_repo,
    get_repo_dir_name,
    retry_clone,
)

logger = logging.getLogger(__name__)


def load_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite", split: str = "test"
):
    from datasets import load_dataset

    data = load_dataset(dataset_name, split=split)
    return {d["instance_id"]: d for d in data}


def load_instance(
    instance_id: str,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
):
    data = load_instances(dataset_name, split=split)
    return data[instance_id]


def sorted_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    sort_by: str = "created_at",
):
    from datasets import load_dataset

    data = load_dataset(dataset_name, split=split)
    instances = list(data)
    instances = sorted(instances, key=lambda x: x[sort_by])
    return instances


def get_repo_dir_name(repo: str):
    return repo.replace("/", "__")


def found_in_expected_spans(instance: dict, spans: dict):
    for file_path, span_ids in instance["expected_spans"].items():
        if not span_ids:
            logging.warning(
                f"{instance['instance_id']} Expected spans for {file_path} is empty"
            )

    missing_spans = get_missing_spans(instance["expected_spans"], spans)
    return not missing_spans


def found_in_alternative_spans(instance: dict, spans: dict):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.info(
                    f"{instance['instance_id']} Alternative spans for {file_path} is empty"
                )

        missing_spans = get_missing_spans(alternative_spans["spans"], spans)
        if not missing_spans:
            return True

    return False


def found_in_alternative_files(instance: dict, files: list):
    if "alternative_spans" not in instance:
        return False
    for alternative_spans in instance["alternative_spans"]:
        for file_path, span_ids in alternative_spans["spans"].items():
            if not span_ids:
                logging.info(
                    f"{instance['instance_id']} Alternative spans for {file_path} is empty"
                )

        missing_spans = get_missing_files(alternative_spans["spans"], files)
        if not missing_spans:
            return True

    return False


def setup_swebench_repo(
    instance_data: Optional[dict] = None,
    instance_id: str = None,
    repo_base_dir: Optional[str] = None,
) -> str:
    assert (
        instance_data or instance_id
    ), "Either instance_data or instance_id must be provided"
    if not instance_data:
        instance_data = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    repo_dir_name = instance_data["repo"].replace("/", "__")
    github_repo_path = f"swe-bench/{repo_dir_name}"
    return setup_github_repo(
        repo=github_repo_path,
        base_commit=instance_data["base_commit"],
        base_dir=repo_base_dir,
    )


def create_repository(
    instance: Optional[dict] = None,
    instance_id: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
    repo_url: Optional[str] = None,
    repo_path: Optional[str] = None,
    commit: Optional[str] = None,
):
    """
    创建代码仓库的工作区。
    支持三种方式：
    1. 通过instance或instance_id创建SWE-bench实例的工作区
    2. 通过repo_url直接克隆Git仓库
    3. 通过repo_path使用已存在的本地仓库

    参数:
        instance: SWE-bench实例数据字典
        instance_id: SWE-bench实例ID
        repo_base_dir: 仓库基础目录
        repo_url: Git仓库URL
        repo_path: 本地仓库路径
        commit: Git提交哈希
    
    返回:
        Repository: 仓库对象
    """
    if repo_path and os.path.exists(repo_path):
        logger.info(f"使用已存在的本地仓库: {repo_path}")
        if commit:
            # 检查提交是否存在
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "cat-file", "-e", commit],
                    cwd=repo_path,
                    capture_output=True,
                    check=True,
                )
                logger.info(f"在本地仓库中找到提交 {commit}")
                return GitRepository(repo_path=repo_path)
            except subprocess.CalledProcessError:
                logger.warning(f"本地仓库 {repo_path} 不包含提交 {commit}")
                # 继续使用提供的URL或实例
        else:
            # 无需检查特定提交
            return GitRepository(repo_path=repo_path)
    
    if repo_url:
        # 直接从URL克隆仓库
        if not repo_base_dir:
            repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")
        
        # 确保目录存在
        os.makedirs(repo_base_dir, exist_ok=True)
        
        # 从URL提取仓库名称
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        
        local_repo_path = f"{repo_base_dir}/{repo_name}"
        
        if os.path.exists(local_repo_path):
            logger.info(f"使用已存在的本地仓库: {local_repo_path}")
            if commit:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["git", "cat-file", "-e", commit],
                        cwd=local_repo_path,
                        capture_output=True,
                        check=True,
                    )
                    logger.info(f"在本地仓库中找到提交 {commit}")
                    return GitRepository(repo_path=local_repo_path)
                except subprocess.CalledProcessError:
                    logger.warning(f"本地仓库 {local_repo_path} 不包含提交 {commit}")
                    shutil.rmtree(local_repo_path)
                    # 继续克隆
            else:
                return GitRepository(repo_path=local_repo_path)
        
        # 克隆仓库
        try:
            import subprocess
            result = subprocess.run(
                ["git", "clone", repo_url, local_repo_path],
                capture_output=True,
                check=True,
            )
            logger.info(f"已克隆 {repo_url} 到 {local_repo_path}")
            
            if commit:
                # 检出特定提交
                result = subprocess.run(
                    ["git", "checkout", commit],
                    cwd=local_repo_path,
                    capture_output=True,
                    check=True,
                )
                logger.info(f"已检出提交 {commit}")
            
            return GitRepository(repo_path=local_repo_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"克隆仓库失败: {e}")
            raise
    
    # 原有逻辑：通过instance或instance_id处理
    assert instance or instance_id, "必须提供instance、instance_id、repo_url或repo_path中的一个"
    if not instance:
        instance = load_instance(instance_id)

    if not repo_base_dir:
        repo_base_dir = os.getenv("REPO_DIR", "/tmp/repos")

    # 确保目录存在
    os.makedirs(os.path.dirname(repo_base_dir), exist_ok=True)
    os.makedirs(repo_base_dir, exist_ok=True)

    repo_dir_name = get_repo_dir_name(instance["repo"])
    local_repo_path = f"{repo_base_dir}/swe-bench_{repo_dir_name}"
    lock_file_path = f"{local_repo_path}.lock"

    # 确保锁文件目录存在
    os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)
    print(repo_base_dir)
    repo_path = f"{repo_base_dir}/swe-bench_{instance['instance_id']}"
    if os.path.exists(repo_path):
        try:
            # 检查提交是否存在
            import subprocess
            result = subprocess.run(
                ["git", "cat-file", "-e", instance["base_commit"]],
                cwd=repo_path,
                capture_output=True,
                check=True,
            )
            logger.info(
                f"在 {repo_path} 找到包含提交 {instance['base_commit']} 的已存在仓库"
            )
            print(repo_path)
            print(instance["base_commit"])
            
            return GitRepository(repo_path=repo_path)
        except subprocess.CalledProcessError:
            logger.warning(
                f"已存在的仓库 {repo_path} 不包含提交 {instance['base_commit']}"
            )
            shutil.rmtree(repo_path)
        except Exception as e:
            logging.warning(f"检查仓库时出错: {e}")
            shutil.rmtree(repo_path)

    with open(lock_file_path, "w") as lock_file:
        logging.debug(f"获取 {local_repo_path} 的锁")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not os.path.exists(local_repo_path):
            # 如果本地仓库不存在，从GitHub克隆
            # github_url = f"https://github.com/swe-bench/{repo_dir_name}.git"
            github_url = f"git@github.com:swe-bench/{repo_dir_name}.git"
            try:
                retry_clone(github_url, local_repo_path)
                logging.info(f"已克隆 {github_url} 到 {local_repo_path}")
            except Exception as e:
                logger.error(f"多次尝试克隆后失败: {e}")
                raise
        logging.debug(f"释放 {local_repo_path} 的锁")
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    # 使用绝对路径而不是相对路径
    absolute_repo_path = os.path.abspath(local_repo_path)
    repo_url = f"file://{absolute_repo_path}"
    print(repo_url)
    return GitRepository.from_repo(
        git_repo_url=repo_url, repo_path=repo_path, commit=instance["base_commit"]
    )


def create_index(
    instance: dict = None,
    repository: Repository | None = None,
    index_store_dir: Optional[str] = None,
    instance_id: Optional[str] = None,
    repo_url: Optional[str] = None,
    repo_path: Optional[str] = None,
    repo_base_dir: Optional[str] = None,
    commit: Optional[str] = None,
    index_name: Optional[str] = None,
    force_rebuild: bool = False,
):
    """
    为仓库创建代码索引。
    支持多种方式指定仓库：
    1. 提供已有的Repository对象
    2. 提供instance或instance_id（SWE-bench实例）
    3. 提供repo_url（Git仓库URL）
    4. 提供repo_path（本地仓库路径）

    参数:
        instance: SWE-bench实例数据字典
        repository: 已初始化的Repository对象
        index_store_dir: 索引存储目录
        instance_id: SWE-bench实例ID
        repo_url: Git仓库URL
        repo_path: 本地仓库路径
        repo_base_dir: 仓库基础目录
        commit: Git提交哈希
        index_name: 索引名称（如果为None，将使用实例ID或从仓库路径/URL派生）
        force_rebuild: 是否强制重建索引
    
    返回:
        CodeIndex: 代码索引对象
    """
    if not index_store_dir: # 默认的索引存储目录
        index_store_dir = os.getenv("MOATLESS_INDEX_DIR", "tmp/index_store")
    
    # 确保索引存储目录存在
    os.makedirs(index_store_dir, exist_ok=True)

    # 如果没有提供repository，则创建一个
    if not repository:
        repository = create_repository(
            instance=instance,
            instance_id=instance_id,
            repo_url=repo_url,
            repo_path=repo_path,
            repo_base_dir=repo_base_dir,
            commit=commit
        )
    
    # 确定索引名称
    if not index_name:
        if instance:
            index_name = instance.get("instance_id")
        elif instance_id:
            index_name = instance_id
        elif repo_path:
            # 从仓库路径派生索引名称
            repo_name = os.path.basename(repo_path)
            if commit:
                index_name = f"{repo_name}_{commit[:7]}"
            else:
                index_name = repo_name
        elif repo_url:
            # 从URL派生索引名称
            repo_name = repo_url.split("/")[-1]
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            if commit:
                index_name = f"{repo_name}_{commit[:7]}"
            else:
                index_name = repo_name
        else:
            # 生成随机索引名称
            import uuid
            index_name = f"repo_index_{uuid.uuid4().hex[:8]}"
    
    persist_dir = os.path.join(index_store_dir, index_name)
    
    # 如果索引已存在且不强制重建，则直接加载
    if os.path.exists(persist_dir) and not force_rebuild:
        logger.info(f"加载已存在的索引 {index_name} 从 {persist_dir}")
        try:
            return CodeIndex.from_persist_dir(persist_dir, file_repo=repository)
        except Exception as e:
            logger.warning(f"加载已存在索引失败: {e}，将尝试从远程下载或重建")
    
    # 如果索引不存在或强制重建，且远程索引url存在，尝试从远程下载
    if not force_rebuild and os.getenv("INDEX_STORE_URL"):
        index_store_url = os.getenv("INDEX_STORE_URL")
        store_url = os.path.join(index_store_url, f"{index_name}.zip")
        try:
            logger.info(f"尝试从 {store_url} 下载已存在的索引 {index_name}")
            return CodeIndex.from_url(store_url, persist_dir, repository)
        except Exception as e:
            logger.warning(f"从远程下载索引失败: {e}，将重建索引")
    
    # 如果无法从远程下载或强制重建，则创建新索引
    logger.info(f"为仓库 {repository.repo_dir} 创建新索引 {index_name}")
    code_index = CodeIndex(file_repo=repository, index_name=index_name)
    print(repository.repo_dir)
    code_index.run_ingestion(repo_path=repository.repo_dir)
    
    # 持久化索引
    logger.info(f"持久化索引到 {persist_dir}")
    code_index.persist(persist_dir)
    
    return code_index
