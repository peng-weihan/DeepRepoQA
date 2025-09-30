import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv

from DeepRepoQA.feedback.feedback_agent import FeedbackAgent

def setup_environment():
    """Setup environment path"""
    current_dir = Path(__file__).parent
    sys.path.append(current_dir)

def initialize_qa_system(repo_path):
    """Initialize QA system"""
    print(f"🔧 Initializing code repository: {repo_path}")
    
    # Import project modules
    from DeepRepoQA.completion.completion import CompletionModel, LLMResponseFormat
    from DeepRepoQA.benchmark.swebench import create_repository
    from DeepRepoQA.index import CodeIndex
    from DeepRepoQA.file_context import FileContext
    from DeepRepoQA.selector import BestFirstSelector
    from DeepRepoQA.feedback import GroundTruthFeedbackGenerator
    from DeepRepoQA.value_function.base import ValueFunction
    from DeepRepoQA.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, FindCalledObject
    from DeepRepoQA.agent.code_qa_agent import CodeQAAgent
    from DeepRepoQA.code_qa.search_tree import CodeQASearchTree

    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_store_dir = os.path.join(base_dir, "dataset/index_store")
    # Construct necessary paths
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Create completion model with configuration from environment variables
    completion_model = CompletionModel(
        # LLM model name (e.g., "gpt-4", "claude-3", etc.)
        model=os.getenv("CUSTOM_LLM_MODEL"),
        # Temperature for response generation (0.0 = deterministic, 1.0 = creative)
        temperature=0.3,
        # Base URL for the LLM API endpoint
        model_base_url=os.getenv("CUSTOM_LLM_API_BASE"),
        # API key for authentication with the LLM service
        model_api_key=os.getenv("CUSTOM_LLM_API_KEY"),
    )
    completion_model.response_format = LLMResponseFormat.TOOLS

    # Create code repository object
    repository = create_repository(repo_path=repo_path)

    # Create index
    print("📚 Creating code index...")
    start_time = time.time()
    code_index = CodeIndex.from_repository(
        repo_path=repo_path,
        index_store_dir=index_store_dir,
        file_repo=repository,
    )
    end_time = time.time()
    print(f"Code index creation time: {end_time - start_time:.2f} seconds")

    # File context
    file_context = FileContext(repo=repository)

    # Selector and Value Function
    selector = BestFirstSelector()
    value_function = ValueFunction(completion_model=completion_model)

    # Action set
    actions = [
        FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
        FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCalledObject(completion_model=completion_model, code_index=code_index, repository=repository),
        SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
        ViewCode(completion_model=completion_model, repository=repository),
        Finish(),
    ]

    # Create Agent
    agent = CodeQAAgent.create(
        repository=repository,
        completion_model=completion_model,
        code_index=code_index,
        preset_actions=actions,
    )
    agent.actions = actions

    # Feedback generator
    feedback_generator = FeedbackAgent(completion_model=completion_model)

    print("✅ System initialization completed!")
    
    return {
        'agent': agent,
        'file_context': file_context,
        'selector': selector,
        'value_function': value_function,
        'feedback_generator': feedback_generator,
        'repository': repository
    }

def ask_question(question, system_components):
    """Process a single question"""
    from DeepRepoQA.code_qa.search_tree import CodeQASearchTree
    
    print(f"\n🤔 Processing question: {question}")
    print("⏳ Please wait...")
    
    # Construct trajectory file path (create unique trajectory file for each question)
    timestamp = int(time.time())
    persist_path = f"tmp/trajectory_{timestamp}.json"
    
    # Create search tree
    search_tree = CodeQASearchTree.create(
        message=question,
        agent=system_components['agent'],
        file_context=system_components['file_context'],
        selector=system_components['selector'],
        value_function=system_components['value_function'],
        feedback_generator=system_components['feedback_generator'],
        max_iterations=15,
        max_expansions=3,
        max_depth=10,
        persist_path=persist_path,
    )

    # Execute search
    node = search_tree.run_search()

    # Output answer
    print("\n" + "="*60)
    print("🔍 Answer:")
    print("="*60)
    print(node.observation.message)
    print("="*60)
    
    return node.observation.message

def interactive_qa_session():
    """Interactive Q&A session"""
    print("🚀 Welcome to the Code Q&A System!")
    print("="*60)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get repository path from environment variable
    # This should point to the local path of the code repository to analyze
    repo_path = os.getenv("REPO_PATH")
    
    # Check if repository path exists
    if not os.path.exists(repo_path):
        print(f"❌ Error: Repository path does not exist: {repo_path}")
        print("Please set the correct REPO_PATH environment variable")
        return
    
    # Initialize system
    system_components = initialize_qa_system(repo_path)
    
    print(f"\n Current code repository: {repo_path}")
    print("💡 You can start asking questions now!")
    print(" Type 'quit' or 'exit' to exit the system")
    print("-" * 60)
    
    while True:
        try:
            # Get user question
            question = input("\n❓ Please enter your question: ").strip()
            
            # Check exit command
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using, goodbye!")
                break
            
            # Check empty input
            if not question:
                print("⚠️  Please enter a valid question")
                continue
            
            # Process question
            start_time = time.time()
            answer = ask_question(question, system_components)
            end_time = time.time()
            
            print(f"\n⏱️  Processing time: {end_time - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\n👋 User interrupted, exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error occurred while processing question: {str(e)}")
            print("Please try again or type 'quit' to exit")

def main():
    """Main function"""
    setup_environment()
    interactive_qa_session()

if __name__ == "__main__":
    main()

