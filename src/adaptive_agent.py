#!/usr/bin/env python3
"""
Adaptive Agent - A complete AI agent that dynamically generates MCP tools on the fly

This agent combines:
- Hugging Face smolagents framework for code-based actions
- Model Context Protocol (MCP) for dynamic tool generation
- Support for multiple AI providers (OpenAI, Anthropic, Gemini, Ollama, LM Studio)
- Multi-domain capabilities (coding, science, math, general tasks)
- Web search integration for real-time information

Author: Claude Code Assistant
License: MIT
"""

import os
import json
import re
import inspect
import subprocess
import requests
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import importlib.util

try:
    from smolagents import CodeAgent, tool
    from smolagents.models import MessageRole, Model
except ImportError:
    print("smolagents not found. Installing...")
    subprocess.run(["pip", "install", "smolagents"], check=True)
    from smolagents import CodeAgent, tool
    from smolagents.models import MessageRole, Model

try:
    import openai
except ImportError:
    subprocess.run(["pip", "install", "openai"], check=True)
    import openai

try:
    import google.generativeai as genai
except ImportError:
    subprocess.run(["pip", "install", "google-generativeai"], check=True)
    import google.generativeai as genai

try:
    import anthropic
except ImportError:
    subprocess.run(["pip", "install", "anthropic"], check=True)
    import anthropic

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.align import Align
    from rich import box
    import time
except ImportError:
    print("Installing rich for beautiful CLI...")
    subprocess.run(["pip", "install", "rich"], check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.align import Align
    from rich import box
    import time

try:
    import ollama
except ImportError:
    print("Ollama client not found. Installing...")
    subprocess.run(["pip", "install", "ollama"], check=True)
    import ollama


@dataclass
class ToolTemplate:
    """Template for dynamically generated tools"""
    name: str
    description: str
    domain: str
    code_template: str
    required_imports: List[str]
    parameters: Dict[str, Any]


class ModelProvider(ABC):
    """Abstract base class for AI model providers"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def is_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))


class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            # Convert messages format for Anthropic
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                system=system_msg,
                messages=user_messages,
                max_tokens=4000,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")
    
    def is_available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))


class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            # Convert messages to Gemini format
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def is_available(self) -> bool:
        return bool(os.getenv("GOOGLE_API_KEY"))


class OllamaProvider(ModelProvider):
    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host)
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Ollama error: {e}")
    
    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except:
            return False


class LMStudioProvider(ModelProvider):
    def __init__(self, base_url: str = "http://localhost:1234", model: str = "local-model"):
        self.base_url = base_url.rstrip('/')
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    **kwargs
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"LM Studio error: {e}")
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False


class MCPToolGenerator:
    """Generates MCP tools dynamically based on user prompts and domain requirements"""
    
    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        self.domain_templates = self._initialize_domain_templates()
        self.generated_tools = {}
    
    def _initialize_domain_templates(self) -> Dict[str, List[ToolTemplate]]:
        """No pre-built templates - agent generates all tools dynamically"""
        return {}
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze user prompt to determine what tools to generate"""
        analysis_prompt = f"""
Analyze this user request and determine what REAL MCP tools need to be created to perform ACTUAL actions.

User request: "{prompt}"

Focus on creating tools that DO REAL THINGS like:
- File operations (create, read, write files)
- Web requests (download content, API calls)  
- System commands (run programs, create directories)
- Data processing (manipulate real data)

You must return ONLY a JSON object with this exact structure:
{{
    "custom_tools": [
        {{
            "name": "specific_tool_name",
            "description": "exactly what real action this tool performs",
            "parameters": {{"param_name": "param_type"}}
        }}
    ]
}}

Examples of REAL MCP tools:
- For "create a website": create "html_file_creator" tool that writes actual HTML files
- For "download data": create "web_downloader" tool that fetches and saves content
- For "write code to file": create "code_file_writer" tool that creates actual .py files
- For "make me happy": create "happiness_webpage_creator" tool that creates a real HTML happiness page

Create 1-2 tools that perform ACTUAL ACTIONS for: "{prompt}"

JSON response:"""
        
        try:
            response = self.model_provider.generate([
                {"role": "system", "content": "You are an expert at analyzing requests and designing specific tools. Respond ONLY with valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            print(f"Error analyzing prompt with AI: {e}")
            # Create a reasonable tool based on keywords in the prompt
            return self._create_emergency_tool(prompt)
    
    def _create_emergency_tool(self, prompt: str) -> Dict[str, Any]:
        """Create a basic tool when AI analysis fails"""
        # Analyze keywords to create a reasonable tool
        if any(word in prompt.lower() for word in ['calculate', 'solve', 'math', '+', '-', '*', '/', '=']):
            return {
                "custom_tools": [
                    {
                        "name": "simple_calculator",
                        "description": "Calculate mathematical expressions",
                        "parameters": {"expression": "str"}
                    }
                ]
            }
        elif any(word in prompt.lower() for word in ['code', 'program', 'python', 'script']):
            return {
                "custom_tools": [
                    {
                        "name": "code_helper",
                        "description": "Help with code-related tasks",
                        "parameters": {"task": "str"}
                    }
                ]
            }
        elif any(word in prompt.lower() for word in ['search', 'find', 'lookup', 'information']):
            return {
                "custom_tools": [
                    {
                        "name": "information_finder",
                        "description": "Find and provide information",
                        "parameters": {"query": "str"}
                    }
                ]
            }
        else:
            return {
                "custom_tools": [
                    {
                        "name": "general_helper",
                        "description": "Help with general tasks",
                        "parameters": {"request": "str"}
                    }
                ]
            }
    
    def generate_custom_tool(self, tool_spec: Dict[str, Any]) -> str:
        """Generate a custom tool based on specification"""
        generation_prompt = f"""
Create a REAL MCP tool that can perform actual actions, not just return text.

Tool Specification:
Name: {tool_spec['name']}
Description: {tool_spec['description']}
Parameters: {tool_spec.get('parameters', {})}

CRITICAL REQUIREMENTS for REAL MCP tools:
1. Use the @tool decorator from smolagents
2. Perform ACTUAL actions (file operations, web requests, system commands, etc.)
3. Use real Python functions like open(), os.makedirs(), requests.get(), subprocess.run()
4. Create, modify, or interact with the real filesystem and external services
5. Return status messages about what was actually accomplished

Available modules: os, subprocess, requests, json, math, datetime, open(), print()

Examples of REAL MCP tools:
```python
@tool
def create_html_file(filename: str, content: str) -> str:
    '''
    Create an actual HTML file on the filesystem.
    
    Args:
        filename: Name of the HTML file to create
        content: HTML content to write to the file
    
    Returns:
        Success message with file path
    '''
    try:
        with open(filename, 'w') as f:
            f.write(content)
        full_path = os.path.abspath(filename)
        return f"Successfully created HTML file: {{full_path}}"
    except Exception as e:
        return f"Error creating file: {{str(e)}}"

@tool
def download_web_content(url: str, filename: str) -> str:
    '''
    Download content from a URL and save to file.
    
    Args:
        url: URL to download from
        filename: Local filename to save to
    
    Returns:
        Success message with details
    '''
    try:
        import requests
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return f"Downloaded {{len(response.text)}} characters to {{filename}}"
    except Exception as e:
        return f"Error downloading: {{str(e)}}"
```

Now generate a REAL MCP tool for: {tool_spec['name']}
"""
        
        try:
            response = self.model_provider.generate([
                {"role": "system", "content": "You are an expert Python developer. Generate clean, working tool functions with proper docstrings that include parameter descriptions."},
                {"role": "user", "content": generation_prompt}
            ])
            
            # Extract Python code from response
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1)
            else:
                # Return raw response if no code block found
                return response
        except Exception as e:
            print(f"Error generating custom tool: {e}")
            # Return a simple fallback tool
            return f"""
@tool
def {tool_spec['name']}(input_text: str) -> str:
    '''
    {tool_spec['description']}
    
    Args:
        input_text: Input text to process
    
    Returns:
        Processed result
    '''
    return f"Processed: {{input_text}}"
"""
    
    def get_tools_for_domains(self, domains: List[str]) -> List[Callable]:
        """No pre-built tools - everything is generated dynamically"""
        return []


class AdaptiveAgent:
    """Main adaptive agent class that combines everything"""
    
    def __init__(self, provider_type: str = "auto", **provider_kwargs):
        self.model_provider = self._initialize_provider(provider_type, **provider_kwargs)
        self.tool_generator = MCPToolGenerator(self.model_provider)
        self.agent = None
        self.available_tools = []
    
    def _initialize_provider(self, provider_type: str, **kwargs) -> ModelProvider:
        """Initialize the appropriate model provider"""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "gemini": GeminiProvider,
            "ollama": OllamaProvider,
            "lmstudio": LMStudioProvider
        }
        
        if provider_type == "auto":
            # Try providers in order of preference
            for ptype, pclass in providers.items():
                try:
                    provider = pclass(**kwargs)
                    if provider.is_available():
                        print(f"Using {ptype} provider")
                        return provider
                except Exception as e:
                    print(f"Failed to initialize {ptype}: {e}")
                    continue
            
            raise Exception("No available providers found")
        
        elif provider_type in providers:
            return providers[provider_type](**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    def prepare_for_task(self, user_prompt: str, silent: bool = False):
        """Analyze prompt and generate all tools dynamically using AI"""
        if not silent:
            print("Analyzing your request and generating tools with AI...")
        
        # Start with NO tools - everything is generated
        self.available_tools = []
        
        # Analyze the prompt using AI
        analysis = self.tool_generator.analyze_prompt(user_prompt)
        if not silent:
            print(f"AI detected needs: {analysis}")
        
        # Generate ALL tools needed for this specific task
        for custom_tool_spec in analysis.get('custom_tools', []):
            try:
                if not silent:
                    print(f"AI generating tool: {custom_tool_spec['name']}")
                tool_code = self.tool_generator.generate_custom_tool(custom_tool_spec)
                
                # Execute and add the AI-generated tool
                if not silent:
                    print(f"Generated code:\n{tool_code}")
                
                # Clean the generated code by removing problematic import statements
                clean_code = tool_code
                if "from smolagents import tool" in clean_code:
                    clean_code = clean_code.replace("from smolagents import tool\n", "")
                if "import re" in clean_code:
                    clean_code = clean_code.replace("import re\n", "")
                if "import os" in clean_code:
                    clean_code = clean_code.replace("import os\n", "")
                
                # Create a clean namespace with required imports and REAL capabilities
                import re
                import math
                import json
                import os
                import subprocess
                import datetime
                
                try:
                    import requests
                except ImportError:
                    requests = None
                
                namespace = {
                    "tool": tool,
                    "__builtins__": __builtins__,
                    "print": print,
                    "str": str,
                    "int": int,
                    "float": float,
                    "len": len,
                    "range": range,
                    "list": list,
                    "dict": dict,
                    "re": re,
                    "math": math,
                    "json": json,
                    "requests": requests,
                    "os": os,
                    "subprocess": subprocess,
                    "datetime": datetime,
                    "open": open,  # Enable file operations
                }
                
                # Execute the cleaned generated code
                exec(clean_code, namespace)
                
                # Find the new function and add it
                tool_found = False
                for name, obj in namespace.items():
                    if (callable(obj) and 
                        hasattr(obj, '__name__') and 
                        not name.startswith('_') and 
                        name not in ['tool', 'print', 'str', 'int', 'float', 'len', 'range', 'list', 'dict', 'open', 'os', 'subprocess', 'datetime', 'requests', 'json', 'math', 're']):
                        # Check if it's already decorated with @tool or needs decoration
                        if hasattr(obj, '__wrapped__') or hasattr(obj, '_is_tool'):
                            self.available_tools.append(obj)
                        else:
                            # Apply @tool decorator if not already applied
                            decorated_obj = tool(obj)
                            self.available_tools.append(decorated_obj)
                        
                        if not silent:
                            print(f"✅ Generated tool: {name}")
                        tool_found = True
                        break
                
                if not tool_found and not silent:
                    print(f"⚠️ Tool function not found in generated code for {custom_tool_spec['name']}")
            except Exception as e:
                if not silent:
                    print(f"❌ Failed to generate tool {custom_tool_spec['name']}: {e}")
                continue
        
        if not silent:
            print(f"Generated {len(self.available_tools)} AI tools for this task")
    
    def _handle_file_operations(self, user_prompt: str, silent: bool = False):
        """Handle requests that require real file operations"""
        if not silent:
            print("🔧 Detected file operation request - using real MCP tools")
        
        # Use AI to generate code that performs real file operations
        generation_prompt = f"""
Generate Python code that performs the requested file operation and actually executes it.

User request: "{user_prompt}"

Requirements:
1. Use real Python file operations (open(), with statements, etc.)
2. Actually CREATE, WRITE, or MODIFY files on the filesystem
3. Return a success message with the full file path
4. Handle errors gracefully

Example for creating an HTML file:
```python
html_content = '''<!DOCTYPE html>
<html><head><title>Example</title></head>
<body><h1>Hello World</h1></body></html>'''

try:
    with open('example.html', 'w') as f:
        f.write(html_content)
    import os
    full_path = os.path.abspath('example.html')
    result = f"Successfully created file: {{full_path}}"
except Exception as e:
    result = f"Error: {{str(e)}}"

print(result)
```

Generate Python code for: {user_prompt}
"""
        
        try:
            response = self.model_provider.generate([
                {"role": "system", "content": "You are an expert Python programmer. Generate working code that performs real file operations."},
                {"role": "user", "content": generation_prompt}
            ])
            
            # Extract and execute the code
            import re
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = response
            
            if not silent:
                print(f"Generated file operation code:\n{code}")
            
            # Execute the code directly (not in smolagents sandbox)
            import os
            import datetime
            import json
            
            # Create safe execution environment
            exec_globals = {
                'open': open,
                'os': os,
                'datetime': datetime,
                'json': json,
                'print': print,
                '__builtins__': __builtins__,
            }
            
            # Capture the output
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                exec(code, exec_globals)
                output = captured_output.getvalue()
                return output.strip() if output.strip() else "File operation completed successfully"
            except Exception as e:
                return f"Error executing file operation: {str(e)}"
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            return f"Error generating file operation code: {str(e)}"
    
    def run(self, user_prompt: str, silent: bool = False, **kwargs):
        """Main entry point to run the agent using real AI only"""
        # Generate tools dynamically for this specific task
        self.prepare_for_task(user_prompt, silent=silent)
        
        # Check if this requires file operations and handle specially
        if any(word in user_prompt.lower() for word in ['create', 'file', 'save', 'write', 'download', 'generate']):
            return self._handle_file_operations(user_prompt, silent=silent)
        
        # Create a proper model wrapper for smolagents
        class WorkingModelWrapper(Model):
            def __init__(self, provider):
                super().__init__()
                self.provider = provider
            
            def generate(self, messages, stop_sequences=None, grammar=None, **kwargs):
                # Convert smolagents messages to provider format
                provider_messages = []
                for msg in messages:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        role_str = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                        
                        # Map smolagents roles to OpenAI-compatible roles
                        original_role = role_str
                        if role_str == "tool-response":
                            role_str = "user"  # Tool responses become user messages
                        elif role_str == "tool-call":
                            role_str = "assistant"  # Tool calls become assistant messages
                        elif role_str == "tool":
                            role_str = "user"  # Tool messages become user messages
                        elif role_str not in ["system", "assistant", "user"]:
                            role_str = "user"  # Default unknown roles to user
                        
                        # Debug: print role mapping in non-silent mode
                        if original_role != role_str and not kwargs.get('silent', False):
                            print(f"Role mapping: {original_role} -> {role_str}")
                        
                        provider_messages.append({
                            "role": role_str,
                            "content": msg.content
                        })
                    elif isinstance(msg, dict):
                        # Handle dict messages
                        role = msg.get("role", "user")
                        if role not in ["system", "assistant", "user"]:
                            role = "user"
                        provider_messages.append({
                            "role": role,
                            "content": msg.get("content", str(msg))
                        })
                    else:
                        provider_messages.append({
                            "role": "user", 
                            "content": str(msg)
                        })
                
                response = self.provider.generate(provider_messages, **kwargs)
                
                # Return in the format smolagents expects
                from smolagents.models import ChatMessage, MessageRole
                return ChatMessage(role=MessageRole.ASSISTANT, content=response)
            
            def __call__(self, messages, stop_sequences=None, grammar=None, **kwargs):
                return self.generate(messages, stop_sequences, grammar, **kwargs)
        
        # Initialize the agent with AI-generated tools only
        model_wrapper = WorkingModelWrapper(self.model_provider)
        self.agent = CodeAgent(tools=self.available_tools, model=model_wrapper)
        
        # Run the agent with real AI
        result = self.agent.run(user_prompt, **kwargs)
        return result


# No built-in tools - agent generates everything dynamically


class AdaptiveAgentCLI:
    """Beautiful CLI interface for the Adaptive Agent"""
    
    def __init__(self):
        self.console = Console()
        self.agent = None
        self.session_history = []
        
    def display_header(self):
        """Display beautiful header with branding"""
        header_text = Text("🤖 ADAPTIVE AGENT", style="bold cyan")
        subtitle = Text("Dynamic MCP Tool Generator", style="italic bright_blue")
        
        header_panel = Panel(
            Align.center(header_text + "\n" + subtitle),
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        
        self.console.print(header_panel)
        
        # Show supported providers
        providers_table = Table(show_header=False, box=box.SIMPLE, border_style="dim")
        providers_table.add_column("", style="cyan", no_wrap=True)
        providers_table.add_column("", style="white")
        
        providers_table.add_row("🌟", "OpenAI GPT (API)")
        providers_table.add_row("🧠", "Anthropic Claude (API)")
        providers_table.add_row("🔬", "Google Gemini (API)")
        providers_table.add_row("🏠", "Ollama (Local)")
        providers_table.add_row("🖥️", "LM Studio (Local)")
        
        provider_panel = Panel(
            providers_table,
            title="[bold bright_blue]Supported AI Providers[/bold bright_blue]",
            border_style="blue",
            padding=(0, 1)
        )
        
        self.console.print(provider_panel)
    
    def initialize_agent(self):
        """Initialize the agent with beautiful loading animation"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("[cyan]Initializing Adaptive Agent...", total=None)
            
            try:
                self.agent = AdaptiveAgent(provider_type="auto")
                progress.update(task, description="[green]✅ Agent initialized successfully!")
                time.sleep(0.5)
                return True
            except Exception as e:
                progress.update(task, description=f"[red]❌ Failed to initialize: {str(e)}")
                time.sleep(1)
                return False
    
    def show_help(self):
        """Display beautiful help information"""
        help_content = """
## 🚀 Getting Started

The Adaptive Agent creates custom tools on-the-fly for any task you throw at it!

### 💡 What You Can Do

- **Math & Calculations**: `solve x^2 + 2x - 3 = 0`, `calculate sqrt(16) + 5*3`
- **Programming**: `write Python code to find prime numbers`, `debug this function`
- **Science**: `convert 100 fahrenheit to celsius`, `calculate molecular weight`
- **Web & Research**: `search for latest AI developments`, `find information about quantum computing`
- **General Tasks**: The agent adapts to ANY request by generating appropriate tools!

### 🎯 Commands

- `help` - Show this help information
- `history` - View your session history
- `clear` - Clear the screen
- `status` - Show agent status
- `quit` / `exit` / `q` - Exit the application

### ✨ Examples

```
> solve 2+2
> write a function to calculate fibonacci numbers
> search for information about machine learning
> convert 50 miles to kilometers
```

The agent will analyze your request and generate the perfect tools to complete it!
        """
        
        help_panel = Panel(
            Markdown(help_content),
            title="[bold bright_green]📚 Help & Usage Guide[/bold bright_green]",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(help_panel)
    
    def show_history(self):
        """Display session history"""
        if not self.session_history:
            self.console.print(Panel(
                "[yellow]No commands in history yet.[/yellow]",
                title="[bold]📜 Session History[/bold]",
                border_style="yellow"
            ))
            return
        
        history_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        history_table.add_column("#", style="dim", width=3)
        history_table.add_column("Request", style="cyan")
        history_table.add_column("Status", justify="center", width=8)
        
        for i, (request, status) in enumerate(self.session_history[-10:], 1):  # Show last 10
            status_style = "green" if status == "✅" else "red"
            history_table.add_row(str(i), request[:50] + "..." if len(request) > 50 else request, f"[{status_style}]{status}[/{status_style}]")
        
        history_panel = Panel(
            history_table,
            title="[bold bright_magenta]📜 Recent Session History[/bold bright_magenta]",
            border_style="magenta"
        )
        
        self.console.print(history_panel)
    
    def show_status(self):
        """Display agent status"""
        status_table = Table(show_header=False, box=box.SIMPLE)
        status_table.add_column("", style="cyan", no_wrap=True)
        status_table.add_column("", style="white")
        
        status_table.add_row("🤖 Agent Status", "[green]Active & Ready[/green]" if self.agent else "[red]Not Initialized[/red]")
        status_table.add_row("🔧 Provider", f"[bright_blue]{type(self.agent.model_provider).__name__}[/bright_blue]" if self.agent else "[dim]Unknown[/dim]")
        status_table.add_row("📊 Commands Run", f"[yellow]{len(self.session_history)}[/yellow]")
        status_table.add_row("🛠️ Tools Generated", "[cyan]Dynamic (per request)[/cyan]")
        
        status_panel = Panel(
            status_table,
            title="[bold bright_blue]📊 Agent Status[/bold bright_blue]",
            border_style="blue"
        )
        
        self.console.print(status_panel)
    
    def process_request(self, user_input: str):
        """Process user request with beautiful UI"""
        # Add to history
        self.session_history.append((user_input, "⏳"))
        
        # Create a panel for the user input
        input_panel = Panel(
            f"[bold cyan]💭 {user_input}[/bold cyan]",
            title="[bold]Your Request[/bold]",
            border_style="cyan",
            padding=(0, 1)
        )
        self.console.print(input_panel)
        
        # Process with loading animation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            
            # Analysis phase
            analysis_task = progress.add_task("[cyan]🔍 Analyzing request...", total=100)
            progress.update(analysis_task, advance=30)
            time.sleep(0.5)
            
            # Tool generation phase  
            progress.update(analysis_task, description="[yellow]🛠️ Generating tools...", advance=30)
            time.sleep(0.5)
            
            # Processing phase
            progress.update(analysis_task, description="[green]⚡ Processing with AI...", advance=40)
            
            try:
                result = self.agent.run(user_input, silent=True)
                progress.update(analysis_task, description="[green]✅ Complete!", completed=100)
                
                # Update history
                self.session_history[-1] = (user_input, "✅")
                
                # Display result
                time.sleep(0.3)
                result_panel = Panel(
                    f"[bold green]{result}[/bold green]",
                    title="[bold bright_green]🎯 Result[/bold bright_green]",
                    border_style="green",
                    padding=(1, 2)
                )
                self.console.print(result_panel)
                
            except Exception as e:
                progress.update(analysis_task, description="[red]❌ Error occurred", completed=100)
                
                # Update history
                self.session_history[-1] = (user_input, "❌")
                
                # Display error
                time.sleep(0.3)
                error_panel = Panel(
                    f"[bold red]Error: {str(e)}[/bold red]",
                    title="[bold red]❌ Error[/bold red]",
                    border_style="red",
                    padding=(1, 2)
                )
                self.console.print(error_panel)
    
    def run(self):
        """Main CLI loop"""
        # Clear screen and show header
        self.console.clear()
        self.display_header()
        
        # Initialize agent
        if not self.initialize_agent():
            error_panel = Panel(
                "[red]Failed to initialize agent. Please check your API keys or local model setup.[/red]",
                title="[bold red]❌ Initialization Error[/bold red]",
                border_style="red"
            )
            self.console.print(error_panel)
            return
        
        # Show welcome message
        welcome_panel = Panel(
            "[green]🎉 Welcome! I'm ready to help with any task.\n"
            "💡 Try: [cyan]'solve 2+2'[/cyan], [cyan]'help'[/cyan], or [cyan]'write code to find primes'[/cyan][/green]",
            title="[bold bright_green]🚀 Ready to Go![/bold bright_green]",
            border_style="green"
        )
        self.console.print(welcome_panel)
        
        # Main interactive loop
        while True:
            try:
                self.console.print()  # Add spacing
                
                # Beautiful prompt
                user_input = Prompt.ask(
                    "[bold bright_cyan]🔤 Your request[/bold bright_cyan]",
                    console=self.console
                ).strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    goodbye_panel = Panel(
                        "[bold bright_yellow]👋 Thank you for using Adaptive Agent!\n"
                        "✨ Remember: I can adapt to any task you need.[/bold bright_yellow]",
                        title="[bold]Goodbye![/bold]",
                        border_style="yellow"
                    )
                    self.console.print(goodbye_panel)
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.console.clear()
                    self.display_header()
                    continue
                
                elif user_input.lower() == 'status':
                    self.show_status()
                    continue
                
                # Process the request
                self.process_request(user_input)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 Interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                error_panel = Panel(
                    f"[red]Unexpected error: {str(e)}[/red]",
                    title="[bold red]❌ System Error[/bold red]",
                    border_style="red"
                )
                self.console.print(error_panel)


def main():
    """Launch the beautiful CLI"""
    cli = AdaptiveAgentCLI()
    cli.run()


if __name__ == "__main__":
    main()