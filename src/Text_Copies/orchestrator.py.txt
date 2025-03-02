import json
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (ensure you have enough memory, and device_map set appropriately)
model_name = "NousResearch/Hermes-3-Llama-3.1-8B"
print("Loading Hermes 3 model... (This may take a minute)")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded. Ready to serve instructions.")

# Define the tools metadata
tools = [
    {
        "name": "create_folder",
        "description": "Create a folder at the specified path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "launch_app",
        "description": "Launch an application by name or full path.",
        "parameters": {
            "type": "object",
            "properties": {
                "app": {"type": "string"}
            },
            "required": ["app"]
        }
    },
    {
        "name": "get_system_info",
        "description": "Retrieve basic system information (OS, CPU, RAM).",
        "parameters": {
            "type": "object",
            "properties": { },
            "required": []
        }
    },
    {
        "name": "type_in_notepad",
        "description": "Open Notepad, type given text, and optionally save to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "filename": {"type": "string"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "calculate",
        "description": "Calculate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
]

# Prepare the system prompt with tools and function call instructions
tool_json = json.dumps(tools)
function_call_schema = json.dumps({
    "properties": {
        "arguments": {"title": "Arguments", "type": "object"},
        "name": {"title": "Name", "type": "string"}
    },
    "required": ["arguments", "name"],
    "title": "FunctionCall",
    "type": "object"
})
system_prompt = (
    "<|im_start|>system\n"
    "You are a function calling AI model. You are provided with function signatures within <tools>"
    + tool_json +
    "</tools>.\n"
    "Use the following JSON schema for each tool call you make:\n"
    + function_call_schema + "\n"
    "For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> tags.\n"
    "Only use a tool if necessary to answer the user’s request. When you have enough information, provide the final answer normally.\n"
    "<|im_end|>\n"
)
# Note: The above prompt is derived from Hermes 3 guidelines ([NousResearch/Hermes-3-Llama-3.1-8B · Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B#:~:text=stock.,name)) but simplified.

# Utility: execute a tool and get output
def execute_tool(name, arguments):
    try:
        if name == "create_folder":
            path = arguments.get("path")
            result = subprocess.run(["powershell", "-File", "scripts/new_folder.ps1", "-Path", path], 
                                    capture_output=True, text=True)
            return result.stdout.strip()
        elif name == "launch_app":
            app = arguments.get("app")
            result = subprocess.run(["powershell", "-File", "scripts/launch_app.ps1", "-App", app], 
                                    capture_output=True, text=True)
            return result.stdout.strip()
        elif name == "get_system_info":
            result = subprocess.run(["powershell", "-File", "scripts/sys_info.ps1"], 
                                    capture_output=True, text=True)
            return result.stdout.strip()
        elif name == "type_in_notepad":
            text = arguments.get("text", "")
            filename = arguments.get("filename", "")
            # Call AutoHotkey script. We assume AutoHotkey.exe is in PATH. If not, specify full path.
            ahk_path = "C:\\Program Files\\AutoHotkey\\AutoHotkey.exe"
            args = [ahk_path, "scripts/notepad_type.ahk", text]
            if filename:
                args.append(filename)
            result = subprocess.run(args, capture_output=True, text=True)
            # The AHK script mostly does GUI actions and doesn't output anything unless error.
            if result.stderr:
                return f"ERROR: {result.stderr.strip()}"
            # Indicate success (no output means likely success)
            return "SUCCESS: Text typed in Notepad" + (f" and saved to {filename}" if filename else "")
        elif name == "calculate":
            expr = arguments.get("expression", "")
            result = subprocess.run(["python", "scripts/calculate.py", expr],
                                    capture_output=True, text=True)
            return result.stdout.strip()
        else:
            return f"ERROR: Unknown tool {name}"
    except Exception as e:
        return f"ERROR: Exception while executing tool: {e}"

print("Hermes 3 Operator is now ready. Type your instructions (or 'exit' to quit).")
while True:
    try:
        user_input = input("\n>> You: ")
    except EOFError:
        break
    if not user_input.strip():
        continue
    if user_input.lower() in ("exit", "quit"):
        print("Exiting Hermes 3 Operator.")
        break

    # Construct conversation with the single user turn (for simplicity, we are not keeping a long chat history here)
    prompt = system_prompt + f"<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=256, stopping_criteria=None, pad_token_id=tokenizer.eos_token_id)
    # Decode only the newly generated part (beyond prompt length)
    prompt_len = inputs['input_ids'].shape[1]
    generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
    # The model's response may contain the tool_call tags or the final answer.
    if "<tool_call>" in generated:
        # Extract JSON from within <tool_call>...</tool_call>
        start_idx = generated.find("<tool_call>")
        end_idx = generated.find("</tool_call>")
        tool_json_str = generated[start_idx+len("<tool_call>"): end_idx].strip()
        print(f"Assistant decided to call a tool with: {tool_json_str}")
        # Parse the JSON
        try:
            tool_request = json.loads(tool_json_str)
        except json.JSONDecodeError as e:
            print("Failed to parse tool_call JSON:", e)
            # If parse fails, just print the raw output and continue
            continue
        tool_name = tool_request.get("name")
        tool_args = tool_request.get("arguments", {})
        # Execute the tool
        result = execute_tool(tool_name, tool_args)
        print(f"[Tool Output: {result}]")
        # Prepare tool response for the model
        tool_response = (
            f"<|im_start|>tool\n<tool_response>{{\"name\": \"{tool_name}\", \"content\": \"{result}\"}}</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # Append tool response to original prompt and ask model to continue
        new_prompt = prompt + generated[:end_idx+len("</tool_call>")] + "\n" + tool_response
        inputs2 = tokenizer(new_prompt, return_tensors="pt").to(model.device)
        output_ids2 = model.generate(**inputs2, max_new_tokens=200, stopping_criteria=None, pad_token_id=tokenizer.eos_token_id)
        final_answer = tokenizer.decode(output_ids2[0][inputs2['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {final_answer.strip()}")
    else:
        # If no tool call, it should be the final answer
        answer = generated.strip()
        # Remove any special tokens or tags just in case
        answer = answer.replace("<|im_end|>", "").replace("<|im_start|>assistant", "")
        print(f"Assistant: {answer}")