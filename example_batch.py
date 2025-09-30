import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
# Add project root directory to Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Load .env file from specified path
env_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"Attempting to load .env file: {env_path}")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ Successfully loaded .env file: {env_path}")
else:
    print(f"❌ .env file not found: {env_path}")
    # Try loading from current working directory
    load_dotenv()
    print("Attempting to load .env file from current working directory")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from DeepRepoQA.completion.completion import CompletionModel
from DeepRepoQA.completion.completion import LLMResponseFormat
from DeepRepoQA.benchmark.swebench import create_repository
from DeepRepoQA.index import CodeIndex
from DeepRepoQA.file_context import FileContext
from DeepRepoQA.selector import BestFirstSelector
from DeepRepoQA.feedback import FeedbackAgent
from DeepRepoQA.value_function.base import ValueFunction
from DeepRepoQA.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish
from DeepRepoQA.agent.code_qa_agent import CodeQAAgent
from DeepRepoQA.agent.code_qa_prompts import *
from DeepRepoQA.code_qa.search_tree import CodeQASearchTree
from DeepRepoQA.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)
import threading
lock = threading.Lock()  # File write lock to prevent concurrent write disorder
def safe_json_serialize(obj):
    """Safely serialize object to JSON-acceptable format"""
    if hasattr(obj, 'model_dump'):
        # Pydantic model
        return obj.model_dump()
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Basic types
        return obj
    elif isinstance(obj, (list, tuple)):
        # List or tuple
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        # Dictionary
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    else:
        # Convert other objects to string
        return str(obj)
# Setup logging

def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mcts_parallel_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
logger = setup_logging()


base_dir = os.path.dirname(os.path.abspath(__file__))
index_store_dir = os.path.join(base_dir, "dataset/index_store")
repo_base_dir = os.path.join(base_dir, "dataset/repos")
persist_path = os.path.join(base_dir, "dataset/trajectory.json")
trajectory_log_dir = os.path.join(base_dir, "logs/trajectories/")
# Configuration constants
BASE_PATHS = {
    "repo_base": repo_base_dir,
    "index_base": index_store_dir, 
    "questions_base": os.path.join(base_dir, "dataset/questions"),
    "answers_base": os.path.join(base_dir, "dataset/answers")
}
# Project configuration
PROJECTS = [
    "flask", 
    # "astropy", 
    # "django", 
    # "matplotlib", 
    # "pylint", 
    # "pytest",
    # "requests",
    # "scikit-learn", 
    # "sphinx", 
    # "sqlfluff", 
    # "sympy", 
    # "xarray"
]

def ensure_trajectory_log_dir():
    """Ensure trajectory log directory exists"""
    os.makedirs(trajectory_log_dir, exist_ok=True)
    return trajectory_log_dir

def log_trajectory(question: str, finish_id: str, trajectory: list, project_name: str = "unknown"):
    """Log trajectory to persistent log file"""
    try:
        ensure_trajectory_log_dir()
        
        # Create log file name (grouped by date and project)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(trajectory_log_dir, f"trajectory_{project_name}_{timestamp}.jsonl")
        
        # Determine if search was successful or failed
        search_status = "success" if finish_id != "failed" else "failed"
        
        # Prepare log data
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "project": project_name,
            "question": question,
            "finish_node_id": finish_id,
            "search_status": search_status,
            "trajectory_length": len(trajectory),
            "trajectory": trajectory
        }
        
        # Thread-safe log writing
        with lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False)
                f.write('\n')
                
        status_emoji = "✅" if search_status == "success" else "⚠️"
        logger.info(f"{status_emoji} Trajectory logged: finish_id={finish_id}, status={search_status}, steps={len(trajectory)}, file={log_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log trajectory: {e}")

def load_data_from_jsonl(path):
    """Load JSONL file data"""
    data_list = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skip invalid JSON line {i+1}: {e}")
                except Exception as e:
                    logger.error(f"Error processing line {i+1}: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to read file {path}: {e}")
        raise
    
    logger.info(f"Successfully loaded {len(data_list)} records from {path}")
    return data_list

def load_existing_answers(output_path):
    """Load existing answers, return set of processed questions"""
    existing_questions = set()
    try:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        question = data.get("question", "")
                        if question and data.get("answer"):
                            existing_questions.add(question)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Found {len(existing_questions)} already processed questions")
        else:
            logger.info("Output file not found, will create new file")
    except Exception as e:
        logger.warning(f"Error reading existing answers: {e}")
    
    return existing_questions

def append_data_to_jsonl(path, data):
    """Thread-safely append data to JSONL file"""
    try:
        with lock:
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        logger.error(f"Failed to write file {path}: {e}")
        raise

def process_single_question(message: str, repo_path: str, index_persist_path: str, project_name: str = "unknown"):
    """Process a single question"""
    logger.info(f"Starting to process question: {message[:100]}...")
    
    try:
        load_dotenv()
        model_name = os.getenv("CUSTOM_LLM_MODEL")
        if not model_name:
            raise ValueError("CUSTOM_LLM_MODEL environment variable not set")
        
        completion_model = CompletionModel(
            model=os.getenv("CUSTOM_LLM_MODEL"),
            temperature=0,
            model_base_url=os.getenv("CUSTOM_LLM_API_BASE"),
            model_api_key=os.getenv("CUSTOM_LLM_API_KEY"),
        )
        
        completion_model.response_format = LLMResponseFormat.TOOLS

        repository = create_repository(repo_path=repo_path, repo_base_dir=repo_base_dir)

        code_index = CodeIndex.from_repository(
            repo_path=repo_path,
            index_store_dir=index_store_dir,
            file_repo=repository
        )
        file_context = FileContext(repo=repository)
        selector = BestFirstSelector()
        value_function = ValueFunction(completion_model=completion_model)

        actions = [
            FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
            FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
            FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
            SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
            ViewCode(completion_model=completion_model, repository=repository),
            Finish(),
        ]

        agent = CodeQAAgent.create(repository=repository, completion_model=completion_model, code_index=code_index, preset_actions=actions)
        feedback_generator = FeedbackAgent(completion_model=completion_model)

        search_tree = CodeQASearchTree.create(
            message=message,
            agent=agent,
            file_context=file_context,
            selector=selector,
            value_function=value_function,
            feedback_generator=feedback_generator,
            max_iterations=os.getenv("MAX_ITERATION"),
            max_expansions=os.getenv("MAX_EXPANSION"),
            max_depth=os.getenv("MAX_DEPTH"),
            persist_path=persist_path,
        )
        result = search_tree.run_search()
        
        # Process return result: could be Node, tuple[Node, str] or None
        if isinstance(result, tuple):
            node, answer = result
            finish_id = node.node_id if node else "failed"
        else:
            node = result
            finish_id = node.node_id if node else "failed"
            answer = node.observation.message if node else "Search failed"
        trajectory = []
        
        # Build trajectory, record executed steps even if search fails
        if node is not None:
            current_node = node
            while current_node is not None:
                if hasattr(current_node, 'action') and current_node.action is not None:
                    try:
                        if hasattr(current_node.action, 'model_dump'):
                            action_data = current_node.action.model_dump()
                            action_data['action_class_name'] = current_node.action.__class__.__name__
                            trajectory.append(action_data)
                        else:
                            action_dict = {}
                            for key, value in current_node.action.__dict__.items():
                                if hasattr(value, 'model_dump'):
                                    action_dict[key] = value.model_dump()
                                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                    action_dict[key] = value
                                else:
                                    action_dict[key] = str(value)
                            action_dict['action_class_name'] = current_node.action.__class__.__name__
                            trajectory.append(action_dict)
                    except Exception as e:
                        logger.warning(f"Failed to serialize action: {e}, using string representation instead")
                        trajectory.append({"action_type": str(type(current_node.action)), "error": str(e)})
                current_node = current_node.parent
            trajectory.reverse()
        else:
            try:
                all_nodes = search_tree.root.get_all_nodes() if hasattr(search_tree, 'root') else []
                for tree_node in all_nodes:
                    if hasattr(tree_node, 'action') and tree_node.action is not None:
                        try:
                            if hasattr(tree_node.action, 'model_dump'):
                                action_data = tree_node.action.model_dump()
                                action_data['action_class_name'] = tree_node.action.__class__.__name__
                                trajectory.append(action_data)
                            else:
                                action_dict = {}
                                for key, value in tree_node.action.__dict__.items():
                                    if hasattr(value, 'model_dump'):
                                        action_dict[key] = value.model_dump()
                                    elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                        action_dict[key] = value
                                    else:
                                        action_dict[key] = str(value)
                                action_dict['action_class_name'] = tree_node.action.__class__.__name__
                                trajectory.append(action_dict)
                        except Exception as e:
                            logger.warning(f"Failed to serialize action: {e}, using string representation instead")
                            trajectory.append({"action_type": str(type(tree_node.action)), "error": str(e)})
            except Exception as e:
                logger.warning(f"Unable to extract trajectory for failed search: {e}")
                trajectory = [{"error": "trajectory_extraction_failed", "message": str(e)}]
        
        log_trajectory(message, finish_id, trajectory, project_name)

        logger.info(f"Question processing completed: {message[:50]}... -> {answer[:50]}...")
        return {
            "question": message,
            "answer": answer
        }
        
    except Exception as e:
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Failed to process question: {message[:50]}... Error: {e}")
        logger.error(f"Full traceback:\n{full_traceback}")
        return {
            "question": message,
            "answer": f"Processing failed: {str(e)}\n\nFull traceback:\n{full_traceback}"
        }


import concurrent.futures

def run_questions_concurrently(input_path: str, output_path: str, repo_path: str, index_persist_path: str, project_name: str = "unknown", max_workers=64):
    """Process a list of questions concurrently"""
    logger.info(f"Starting concurrent processing, input: {input_path}, output: {output_path}")
    
    try:
        data_list = load_data_from_jsonl(input_path)
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return
        
    if not data_list:
        logger.warning("No data to process")
        return
    
    # Load existing answers
    existing_questions = load_existing_answers(output_path)
    
    # Filter out already processed questions
    filtered_data_list = []
    skipped_count = 0
    for data in data_list:
        question = data.get("question", "")
        if question in existing_questions:
            skipped_count += 1
            logger.info(f"⏭️ Skipped already processed question: {question[:50]}...")
        else:
            filtered_data_list.append(data)
    
    # Detailed statistics
    total_read = len(data_list)
    total_filtered = skipped_count
    total_remaining = len(filtered_data_list)
    
    logger.info(f"📊 Data processing statistics:")
    logger.info(f"  - Total questions read: {total_read}")
    logger.info(f"  - Existing answers: {total_filtered}")
    logger.info(f"  - Remaining questions: {total_remaining}")
    logger.info(f"  - Filter ratio: {total_filtered/total_read*100:.1f}%" if total_read > 0 else "  - Filter ratio: 0%")
    
    if not filtered_data_list:
        logger.info("All questions have been processed, no need to repeat")
        return
        
    durations = []
    success_count = 0
    error_count = 0

    def task(data):
        nonlocal success_count, error_count
        
        question = data.get("question", "")
        if not question:
            logger.warning("No 'question' field in data, skipped")
            return 
        
        start_time = time.perf_counter()
        try:
            res = process_single_question(question, repo_path, index_persist_path, project_name)
            data["answer"] = res.get("answer", "No answer")
            data["processed_at"] = datetime.now().isoformat()
            append_data_to_jsonl(output_path, data)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            durations.append(duration)
            success_count += 1

            logger.info(f"✅ Completed question: {question[:50]}... (Duration: {duration:.2f}s)")

        except Exception as e:
            import traceback
            end_time = time.perf_counter()
            duration = end_time - start_time
            durations.append(duration)
            error_count += 1
            
            full_traceback = traceback.format_exc()
            logger.error(f"❌ Failed to process: {question[:50]}... Error: {e}")
            logger.error(f"Full traceback:\n{full_traceback}")
            
            # Record failed data
            data["answer"] = f"Processing failed: {str(e)}\n\nFull traceback:\n{full_traceback}"
            data["processed_at"] = datetime.now().isoformat()
            data["error"] = full_traceback
            try:
                append_data_to_jsonl(output_path, data)
            except Exception as write_error:
                logger.error(f"Error writing failed data: {write_error}")
    
    logger.info(f"Using {max_workers} worker threads to process {len(filtered_data_list)} questions")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, data) for data in filtered_data_list]
        concurrent.futures.wait(futures)

    # Statistics
    if durations:
        avg_duration = sum(durations) / len(durations)
        total_duration = sum(durations)
        logger.info(f"📊 Processing statistics:")
        logger.info(f"  - Total tasks: {len(durations)}")
        logger.info(f"  - Success: {success_count}")
        logger.info(f"  - Failed: {error_count}")
        logger.info(f"  - Average duration: {avg_duration:.2f}s")
        logger.info(f"  - Total duration: {total_duration:.2f}s")
    else:
        logger.warning("No task durations recorded")

def create_project_config(project_name: str, enabled: bool = False) -> dict:
    """Create project configuration"""
    return {
        "enabled": enabled,
        "repo_path": f"{BASE_PATHS['repo_base']}/{project_name}",
        "index_persist_path": f"{BASE_PATHS['index_base']}/{project_name}",
        "input_jsonl": f"{BASE_PATHS['questions_base']}/{project_name}.jsonl",
        "output_jsonl": f"{BASE_PATHS['answers_base']}/{project_name}.jsonl"
    }

def get_enabled_projects():
    """Get enabled project configurations"""
    # Currently enabled projects
    enabled_projects = PROJECTS 

    return [
        create_project_config(project, project in enabled_projects)
        for project in PROJECTS
    ]

def validate_paths(config: dict) -> bool:
    """Validate whether paths exist"""
    required_paths = ["repo_path","input_jsonl"]
    # required_paths = ["repo_path", "index_persist_path", "input_jsonl"]
    for path_key in required_paths:
        if not os.path.exists(config[path_key]):
            print(f"❌ Path does not exist: {config[path_key]}")
            return False
    return True

def ensure_output_dir(output_path: str):
    """Ensure the output directory exists"""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)


def analyze_trajectory_completeness(project_name: str):
    """Analyze the completeness of trajectory records"""
    try:
        # Build file paths
        timestamp = datetime.now().strftime("%Y%m%d")
        trajectory_file = os.path.join(trajectory_log_dir, f"trajectory_{project_name}_{timestamp}.jsonl")
        questions_file = f"{BASE_PATHS['questions_base']}/{project_name}.jsonl"
        answers_file = f"{BASE_PATHS['answers_base']}/{project_name}.jsonl"
        
        # Count lines in each file
        trajectory_count = 0
        questions_count = 0
        answers_count = 0
        success_count = 0
        failed_count = 0
        
        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        trajectory_count += 1
                        if data.get("search_status") == "success":
                            success_count += 1
                        else:
                            failed_count += 1
                    except json.JSONDecodeError:
                        continue
        
        if os.path.exists(questions_file):
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_count = sum(1 for line in f if line.strip())
        
        if os.path.exists(answers_file):
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_count = sum(1 for line in f if line.strip())
        
        # Output statistics
        logger.info(f"📊 {project_name} Trajectory Completeness Analysis:")
        logger.info(f"  - Total questions: {questions_count}")
        logger.info(f"  - Total answers: {answers_count}")
        logger.info(f"  - Total trajectories: {trajectory_count}")
        logger.info(f"  - Successful searches: {success_count}")
        logger.info(f"  - Failed searches: {failed_count}")
        
        if questions_count > 0:
            completion_rate = trajectory_count / questions_count * 100
            success_rate = success_count / questions_count * 100 if questions_count > 0 else 0
            logger.info(f"  - Trajectory completion rate: {completion_rate:.1f}%")
            logger.info(f"  - Search success rate: {success_rate:.1f}%")
        
        return {
            "questions": questions_count,
            "answers": answers_count,
            "trajectories": trajectory_count,
            "success": success_count,
            "failed": failed_count
        }
        
    except Exception as e:
        logger.error(f"Error analyzing trajectory completeness: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting parallel MCTS processing...")
    
    # Get project configurations
    project_configs = get_enabled_projects()
    enabled_configs = [config for config in project_configs if config["enabled"]]
    
    if not enabled_configs:
        print("❌ No enabled project configurations")
        exit(1)
    
    print(f"📋 Preparing to process {len(enabled_configs)} projects")
    
    total_start_time = time.time()
    
    for i, config in enumerate(enabled_configs, 1):
        project_name = os.path.basename(config["repo_path"])
        print(f"\n[{i}/{len(enabled_configs)}] 🔄 Processing project: {project_name}")
        
        # Validate paths
        if not validate_paths(config):
            print(f"⚠️ Skipping project {project_name}: path validation failed")
            continue
            
        # Ensure output directory exists
        ensure_output_dir(config["output_jsonl"])
        
        # Process project
        start_time = time.time()
        try:
            run_questions_concurrently(
                input_path=config["input_jsonl"], 
                output_path=config["output_jsonl"], 
                repo_path=config["repo_path"], 
                index_persist_path=config["index_persist_path"],
                project_name=project_name,
                max_workers=int(os.getenv("MAX_WORKERS"))
            )
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ Project {project_name} completed, duration: {duration:.2f} seconds")
            
            # Analyze trajectory completeness
            analyze_trajectory_completeness(project_name)
            
        except Exception as e:
            print(f"❌ Project {project_name} failed: {e}")
            continue
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n🎉 All projects processed! Total duration: {total_duration:.2f} seconds")
