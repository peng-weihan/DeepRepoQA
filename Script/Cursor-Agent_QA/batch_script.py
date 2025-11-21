from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from subprocess import Popen, PIPE
import rich
import json
import threading
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import subprocess
load_dotenv()

class TestResult(BaseModel):
    question: str
    answer: str
    trajectory: str
    latency: float = 0.0  # Latency in seconds
    input_tokens: int = 0  # Number of input tokens
    output_tokens: int = 0  # Number of output tokens
    total_tokens: int = 0  # Total number of tokens

def run_cmd(cmd: str, output_format: str = "text") -> tuple[str, float, dict]:
    """Execute command and print output in real-time, format: (Thread i): output
    Returns: (output content, latency time, token info dict)"""
    import select
    import time
    
    thread_id = threading.current_thread().ident
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    output_lines = []
    stderr_lines = []
    last_output_time = time.time()
    timeout_seconds = 120  # Timeout if no new output for 120 seconds
    token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    try:
        while True:
            # Check if stdout has readable data, non-blocking
            if select.select([process.stdout], [], [], 1.0)[0]:  # 1 second timeout check
                line = process.stdout.readline()
                if line:
                    line = line.rstrip('\n')
                    output_lines.append(line)
                    rich.print(f"[blue](Thread {thread_id})[/blue]: {line}", flush=True)
                    last_output_time = time.time()  # Update last output time
                else:
                    # readline returned empty, process may have ended
                    if process.poll() is not None:
                        break
            
            # Check if stderr has readable data (may contain token info)
            if select.select([process.stderr], [], [], 0.1)[0]:
                line = process.stderr.readline()
                if line:
                    stderr_lines.append(line.rstrip('\n'))
            
            # Check if process has ended
            if process.poll() is not None:
                # Read remaining output
                remaining = process.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        if line:
                            output_lines.append(line)
                            rich.print(f"[blue](Thread {thread_id})[/blue]: {line}", flush=True)
                # Read remaining stderr
                remaining_stderr = process.stderr.read()
                if remaining_stderr:
                    stderr_lines.extend(remaining_stderr.splitlines())
                break
            # Check if final answer is found
            if "<end_of_answer>" in ''.join(output_lines):
                process.kill()
                process.wait()
                break
            # Check if timeout (120 seconds without new output)
            if time.time() - last_output_time > timeout_seconds:
                rich.print(f"[red](Thread {thread_id}) No output for {timeout_seconds}s - killing process[/red]", flush=True)
                process.kill()
                process.wait()  # Wait for process cleanup
                break
            

        
        # Wait for process to complete
        return_code = process.returncode
        if return_code != 0:
            rich.print(f"[yellow](Thread {thread_id}) Process exited with code {return_code}[/yellow]", flush=True)
        
        # Try to parse token info from stderr or output
        all_output = '\n'.join(output_lines + stderr_lines)
        extracted_text = '\n'.join(output_lines)  # Used to extract text content
        full_trajectory = '\n'.join(output_lines)  # Save complete trajectory (including all output)
        
        if output_format in ["json", "stream-json"]:
            # If JSON format, try to parse JSON
            json_text_parts = []
            json_objects = []  # Save all JSON objects for building complete trajectory
            for line in output_lines:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        # Save complete JSON object
                        json_objects.append(data)
                        
                        # Extract text content - handle cursor-agent's actual JSON format
                        if "result" in data:
                            json_text_parts.append(str(data["result"]))
                        elif "content" in data:
                            json_text_parts.append(str(data["content"]))
                        elif "text" in data:
                            json_text_parts.append(str(data["text"]))
                        elif "message" in data:
                            msg = data["message"]
                            if isinstance(msg, dict) and "content" in msg:
                                json_text_parts.append(str(msg["content"]))
                            elif isinstance(msg, str):
                                json_text_parts.append(msg)
                        elif "response" in data:
                            json_text_parts.append(str(data["response"]))
                        # Check if there is tool call information
                        elif "type" in data:
                            # If tool call type, save detailed information
                            if data.get("type") in ["tool_call", "function_call", "action"]:
                                json_text_parts.append(f"[Tool Call: {data.get('name', 'unknown')}] {json.dumps(data, indent=2)}")
                            elif data.get("type") == "result" and "result" in data:
                                json_text_parts.append(str(data["result"]))
                        
                        # Check if there is token info in JSON - check all possible fields
                        if "usage" in data:
                            usage = data["usage"]
                            token_info["input_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                            token_info["output_tokens"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
                            token_info["total_tokens"] = usage.get("total_tokens", 0)
                        elif "tokens" in data:
                            tokens = data["tokens"]
                            if isinstance(tokens, dict):
                                token_info["input_tokens"] = tokens.get("input", tokens.get("prompt", tokens.get("prompt_tokens", 0)))
                                token_info["output_tokens"] = tokens.get("output", tokens.get("completion", tokens.get("completion_tokens", 0)))
                                token_info["total_tokens"] = tokens.get("total", tokens.get("total_tokens", 0))
                        elif "prompt_tokens" in data or "completion_tokens" in data:
                            token_info["input_tokens"] = data.get("prompt_tokens", data.get("input_tokens", 0))
                            token_info["output_tokens"] = data.get("completion_tokens", data.get("output_tokens", 0))
                            token_info["total_tokens"] = data.get("total_tokens", 0)
                        # Check nested fields
                        elif "metadata" in data and isinstance(data["metadata"], dict):
                            metadata = data["metadata"]
                            if "usage" in metadata:
                                usage = metadata["usage"]
                                token_info["input_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                                token_info["output_tokens"] = usage.get("completion_tokens", usage.get("output_tokens", 0))
                                token_info["total_tokens"] = usage.get("total_tokens", 0)
                        elif "stats" in data and isinstance(data["stats"], dict):
                            stats = data["stats"]
                            token_info["input_tokens"] = stats.get("prompt_tokens", stats.get("input_tokens", 0))
                            token_info["output_tokens"] = stats.get("completion_tokens", stats.get("output_tokens", 0))
                            token_info["total_tokens"] = stats.get("total_tokens", 0)
                except json.JSONDecodeError:
                    # If not JSON, keep original text
                    json_text_parts.append(line)
            
            # If text was extracted from JSON, use extracted text
            if json_text_parts:
                extracted_text = '\n'.join(json_text_parts)
            
            # Build complete trajectory: save formatted version of all JSON objects
            if json_objects:
                full_trajectory = json.dumps(json_objects, indent=2, ensure_ascii=False)
            else:
                full_trajectory = '\n'.join(output_lines)
        
        # If JSON parsing failed or no token info found, try to parse from text
        if token_info["total_tokens"] == 0:
            token_info = parse_token_info(all_output)
            # If still not found, try to parse from stderr (some tools may output token info to stderr)
            if token_info["total_tokens"] == 0 and stderr_lines:
                stderr_output = '\n'.join(stderr_lines)
                stderr_token_info = parse_token_info(stderr_output)
                if stderr_token_info["total_tokens"] > 0:
                    token_info = stderr_token_info
            
    except Exception as e:
        rich.print(f"[red](Thread {thread_id}) ERROR: {str(e)}[/red]", flush=True)
        try:
            process.kill()
        except:
            pass
        latency = time.time() - start_time
        return (f"Error: {str(e)}", latency, token_info)
    
    # Calculate latency
    latency = time.time() - start_time
    # Return (extracted text content for extracting answer, complete trajectory, latency, token info)
    return (extracted_text, full_trajectory, latency, token_info)

def parse_token_info(trajectory: str) -> dict:
    """Parse token info from output
    Try to match common token statistics formats"""
    import re
    token_info = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    # Try to match various possible token formats
    patterns = [
        (r'total[_\s]*tokens?[:\s]+(\d+)', 'total_tokens'),
        (r'input[_\s]*tokens?[:\s]+(\d+)', 'input_tokens'),
        (r'output[_\s]*tokens?[:\s]+(\d+)', 'output_tokens'),
        (r'prompt[_\s]*tokens?[:\s]+(\d+)', 'input_tokens'),
        (r'completion[_\s]*tokens?[:\s]+(\d+)', 'output_tokens'),
        (r'"total_tokens":\s*(\d+)', 'total_tokens'),
        (r'"prompt_tokens":\s*(\d+)', 'input_tokens'),
        (r'"completion_tokens":\s*(\d+)', 'output_tokens'),
    ]
    
    for pattern, key in patterns:
        matches = re.findall(pattern, trajectory, re.IGNORECASE)
        if matches:
            try:
                token_info[key] = int(matches[-1])  # Take the last match
            except ValueError:
                pass
    
    # If total_tokens not found, try to calculate
    if token_info["total_tokens"] == 0 and (token_info["input_tokens"] > 0 or token_info["output_tokens"] > 0):
        token_info["total_tokens"] = token_info["input_tokens"] + token_info["output_tokens"]
    
    return token_info

def load_processed_questions(output_file: str) -> set[str]:
    """Load set of processed questions from output file"""
    processed_questions = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "question" in data:
                                processed_questions.add(data["question"])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            rich.print(f"[yellow]Warning: Failed to load processed questions from {output_file}: {e}[/yellow]")
    return processed_questions

def generate_trajactory(questions: List[str], repo_path: str, model: str = "auto", max_concurrency: int = os.cpu_count(), output_file: str = None, output_format: str = "json", cursor_agent_path: str = None) -> List[TestResult]:
    results = []
    cursor_api_key = os.getenv("CURSOR_API_KEY")
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    # Determine cursor-agent path: prioritize configured full path, otherwise use PATH
    if cursor_agent_path and os.path.exists(cursor_agent_path):
        cursor_agent_cmd = cursor_agent_path
    else:
        cursor_agent_cmd = "cursor-agent"
        if cursor_agent_path:
            rich.print(f"[yellow]Warning: Specified cursor-agent path '{cursor_agent_path}' not found, using PATH[/yellow]")
    
    # Create list of (command, question) pairs
    cmd_question_pairs = []
    prompt_template = """
You are an expert code analyst. Your task is reading and searching through code repositories and answer questions.

Task Details:  
1. Search and analyze only the code, documentation, and knowledge inside this repository 
related to the following question.  
2. Do not use or assume external knowledge unless explicitly required.  
3. Summarize findings into a concise, direct answer strictly based on repository content.  
4. Output only the final synthesized answer, without code snippets, explanations, or commentary.
5. Your tool calls should be concise and focused. Limits up to 10 tool calls before you must summarize your findings.
6. IMPORTANT: You must respond in English only. Do not use Chinese or any other language.

When you have sufficient information to provide a reasonable answer (even if not perfect), provide your final answer wrapped in:

<start_of_answer>
Your answer here
<end_of_answer>


Then stop furthur analysis, finish the task.
Note: Provide the final answer concisely and directly, without code snippets, extra explanations or commentary. Always respond in English.

Now the code repo at {repo_path}. Gather info from it, and answer my question {question}.
    """
    
    for question in questions:
        prompt = prompt_template.format(repo_path=repo_path, question=question)
        cursor_cmd = f"""{cursor_agent_cmd} --api-key={cursor_api_key} -p "{prompt}" --model "{model}" --output-format {output_format}"""
        cmd_question_pairs.append((cursor_cmd, question))
    
    # Use as_completed to avoid blocking
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Submit all tasks
        future_to_question = {
            executor.submit(run_cmd, cmd, output_format): question 
            for cmd, question in cmd_question_pairs
        }
        
        # Process completed tasks
        for future in tqdm(as_completed(future_to_question), total=len(questions), desc="Processing questions"):
            question = future_to_question[future]
            try:
                extracted_text, full_trajectory, latency, token_info = future.result()
                total_latency += latency
                
                # Use token info parsed from command output
                total_input_tokens += token_info["input_tokens"]
                total_output_tokens += token_info["output_tokens"]
                total_tokens += token_info["total_tokens"]
                
                # Extract final answer (from extracted text)
                result = None
                if "<start_of_answer>" in extracted_text and "<end_of_answer>" in extracted_text:
                    final_answer = extracted_text.split("<start_of_answer>")[1].split("<end_of_answer>")[0].strip()
                    result = TestResult(
                        question=question,
                        answer=final_answer,
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                # HINT: Sometimes there may be errors with multiple `/`, but such tags are intentionally not in XML format to bypass cursor-agent's mysterious rate limiting and review
                elif "<start_of_answer>" in extracted_text and "</end_of_answer>" in extracted_text:
                    final_answer = extracted_text.split("<start_of_answer>")[1].split("</end_of_answer>")[0].strip()
                    result = TestResult(
                        question=question,
                        answer=final_answer,
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                elif "<start_of_answer>" in extracted_text:
                    final_answer = extracted_text.split("<start_of_answer>")[1].strip()
                    result = TestResult(
                        question=question,
                        answer=final_answer,
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                else:
                    result = TestResult(
                        question=question,
                        answer="No answer found",
                        trajectory=full_trajectory,  # Save complete trajectory
                        latency=latency,
                        input_tokens=token_info["input_tokens"],
                        output_tokens=token_info["output_tokens"],
                        total_tokens=token_info["total_tokens"],
                    )
                
                results.append(result)
                # Write to file immediately
                if output_file:
                    with open(output_file, "a") as f:
                        f.write(result.model_dump_json())
                        f.write("\n")
            except Exception as e:
                result = TestResult(
                    question=question,
                    answer=f"Error: {str(e)}",
                    trajectory="",
                    latency=0.0,
                )
                results.append(result)
                # Write to file immediately
                if output_file:
                    with open(output_file, "a") as f:
                        f.write(result.model_dump_json())
                        f.write("\n")
    
    # Print statistics
    avg_latency = total_latency / len(questions) if len(questions) > 0 else 0
    rich.print(f"\n[bold green]Summary:[/bold green] {len(results)}/{len(questions)} questions answered")
    rich.print(f"[bold cyan]Latency:[/bold cyan] Total: {total_latency:.2f}s, Average: {avg_latency:.2f}s")
    if total_tokens > 0:
        rich.print(f"[bold cyan]Token Usage:[/bold cyan] Input: {total_input_tokens:,}, Output: {total_output_tokens:,}, Total: {total_tokens:,}")
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file (JSON format)")
    parser.add_argument("--repo-base-dir", type=str, default=None, help="Path to the repository")
    parser.add_argument("--question-dir", type=str, default=None, help="Directory containing question files")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store output files")
    parser.add_argument("--model", type=str, default=None, help="auto")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum number of concurrent processes")
    parser.add_argument("--repo-filter", type=str, default=None, help="Only process specific repo (e.g., 'conan')")
    parser.add_argument("--output-format", type=str, default="json", choices=["text", "json", "stream-json"], help="Output format for cursor-agent (text, json, or stream-json)")
    parser.add_argument("--cursor-agent-path", type=str, default=None, help="Full path to cursor-agent executable (locks version)")
    args = parser.parse_args()
    
    # If config is a relative path, search in script directory
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    # Read parameters from config file
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        rich.print(f"[green]Loaded config from {config_path}[/green]")
    else:
        rich.print(f"[yellow]Config file {config_path} not found, using defaults[/yellow]")
    
    # Command line arguments take priority over config file
    repo_base_dir = args.repo_base_dir or config.get("repo_base_dir", "repos")
    question_dir = args.question_dir or config.get("question_dir", "/path/to/questions")
    output_dir = args.output_dir or config.get("output_dir", "cursor-questions/results")
    model = args.model or config.get("model", "auto")
    max_concurrency = args.max_concurrency or config.get("max_concurrency", 4)
    repo_filter = args.repo_filter or config.get("repo_filter", None)
    output_format = args.output_format or config.get("output_format", "json")
    cursor_agent_path = args.cursor_agent_path or config.get("cursor_agent_path", None)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    repos: list[str] = []
    questions: list[list[str]] = []
    ground_truths: list[list[str]] = []
    for question_file in os.listdir(question_dir):
        # If repository filter specified, only process matching files
        if repo_filter and not question_file.startswith(f"{repo_filter}.jsonl"):
            continue
        with open(os.path.join(question_dir, question_file), "r") as f:
            lines = [line.strip() for line in f]
            questions.append([json.loads(line)["question"] for line in lines])
            ground_truths.append([json.loads(line)["ground_truth"] for line in lines])
            repos.append(
                 question_file.split(".")[0]
            )
    # Overall statistics
    all_total_latency = 0.0
    all_total_input_tokens = 0
    all_total_output_tokens = 0
    all_total_tokens = 0
    all_total_questions = 0
    
    for repo, question_list, ground_truth_list in tqdm(zip(repos, questions, ground_truths), desc="Processing repos"):
        # Build complete repository path
        if os.path.isdir(repo_base_dir):
            # repo_base_dir is base directory, combine with repo name
            repo_path = os.path.join(repo_base_dir, repo)
            # If combined path doesn't exist, try using repo_base_dir directly (might be full path)
            if not os.path.exists(repo_path):
                repo_path = repo_base_dir if os.path.exists(repo_base_dir) else repo
        else:
            # repo_base_dir is not a directory, might be full path or doesn't exist
            repo_path = repo_base_dir if os.path.exists(repo_base_dir) else repo
        
        rich.print(f"[cyan]Processing repo: {repo_path}[/cyan]")
        output_file_path = os.path.join(output_dir, f"{repo}.jsonl")
        
        # Load processed questions
        processed_questions = load_processed_questions(output_file_path)
        if processed_questions:
            rich.print(f"[yellow]Found {len(processed_questions)} already processed questions, filtering...[/yellow]")
        
        # Filter out processed questions while maintaining correspondence between questions and ground_truth
        filtered_questions = []
        filtered_ground_truths = []
        skipped_count = 0
        for q, gt in zip(question_list, ground_truth_list):
            if q not in processed_questions:
                filtered_questions.append(q)
                filtered_ground_truths.append(gt)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            rich.print(f"[green]Skipped {skipped_count} already processed questions, {len(filtered_questions)} remaining[/green]")
        
        if len(filtered_questions) == 0:
            rich.print(f"[yellow]All questions for {repo} have been processed, skipping...[/yellow]")
            all_total_questions += len(question_list)
            continue
        
        test_results = generate_trajactory(filtered_questions, repo_path, model=model, max_concurrency=max_concurrency, output_file=output_file_path, output_format=output_format, cursor_agent_path=cursor_agent_path)
        
        # Accumulate statistics (including newly processed and skipped)
        for r in test_results:
            all_total_latency += r.latency
            all_total_input_tokens += r.input_tokens
            all_total_output_tokens += r.output_tokens
            all_total_tokens += r.total_tokens
        all_total_questions += len(question_list)  # Count all questions, including skipped
        
        rich.print(f"Results saved to {output_file_path}")
    
    # Print overall statistics
    if all_total_questions > 0:
        avg_latency = all_total_latency / all_total_questions
        rich.print(f"\n[bold yellow]=== Overall Statistics ===[/bold yellow]")
        rich.print(f"[bold]Total Questions:[/bold] {all_total_questions}")
        rich.print(f"[bold]Total Latency:[/bold] {all_total_latency:.2f}s")
        rich.print(f"[bold]Average Latency:[/bold] {avg_latency:.2f}s per question")
        if all_total_tokens > 0:
            rich.print(f"[bold]Total Token Usage:[/bold] Input: {all_total_input_tokens:,}, Output: {all_total_output_tokens:,}, Total: {all_total_tokens:,}")
    