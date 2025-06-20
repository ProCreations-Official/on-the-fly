# 🤖 On-The-Fly Adaptive Agent

[![GitHub Stars](https://img.shields.io/github/stars/ProCreations-Official/on-the-fly?style=for-the-badge&logo=github&color=yellow)](https://github.com/ProCreations-Official/on-the-fly/stargazers)
[![License](https://img.shields.io/github/license/ProCreations-Official/on-the-fly?style=for-the-badge&color=blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Rich CLI](https://img.shields.io/badge/CLI-Rich%20%26%20Beautiful-brightgreen?style=for-the-badge)](https://github.com/Textualize/rich)

A **revolutionary AI agent** that dynamically generates MCP (Model Context Protocol) tools on-the-fly for any task. Features a **beautiful, modern CLI interface** with rich colors, progress bars, and elegant panels.

> 🎯 **No hardcoded tools** - Everything is generated fresh using real AI!  
> ⚡ **Multi-provider support** - Works with any AI model  
> 🎨 **Beautiful interface** - Modern terminal UI experience

## ✨ Features

- **🎨 Beautiful CLI Interface**: Sleek, modern terminal UI with rich colors and animations
- **🛠️ Dynamic Tool Generation**: Creates custom tools based on user prompts using real AI
- **🌐 Multi-Provider Support**: Works with OpenAI GPT, Anthropic Claude, Google Gemini, Ollama, and LM Studio
- **🔬 Multi-Domain Capabilities**: Handles coding, mathematics, science, web search, and general tasks
- **⚡ MCP Integration**: Uses Model Context Protocol for standardized tool interactions
- **🧠 Code-Based Actions**: Uses smolagents' code agent approach for better composability
- **🔍 Auto-Detection**: Automatically detects the best available AI provider
- **📊 Session History**: Track your requests and results with beautiful tables
- **🎯 Real-Time Progress**: Live progress indicators for tool generation and processing

## 🚀 Quick Start

### Prerequisites

Set up at least one AI provider:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-api-key"

# For Ollama (local)
# Install Ollama and pull a model: ollama pull llama2

# For LM Studio (local)
# Install LM Studio and start the local server
```

### Installation & Usage

```bash
# Launch the beautiful CLI interface
python3 adaptive_agent.py

# Or test the CLI components
python3 test_cli_basic.py

# Run a CLI demo
python3 demo_cli.py
```

![CLI Interface Features](https://img.shields.io/badge/CLI-Rich%20%26%20Beautiful-brightgreen)
![Real-time Progress](https://img.shields.io/badge/Progress-Live%20Updates-blue)
![Multi-domain](https://img.shields.io/badge/Domains-Math%2C%20Code%2C%20Science-orange)

## 🎯 Usage Examples

The agent automatically detects your task type and generates appropriate tools:

### Mathematics
```
> Solve the equation x^2 + 2x - 3 = 0
```

### Coding
```
> Write Python code to find all prime numbers up to 100
```

### Science
```
> Convert 100 fahrenheit to celsius
```

### Web Search
```
> Search for the latest news about quantum computing
```

### Multi-Domain
```
> Plot the function sin(x) + cos(2x) from -10 to 10 and analyze its properties
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Adaptive Agent                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│  │   User Prompt   │ -> │ Prompt Analyzer │ -> │ Tool Generator  ││
│  └─────────────────┘    └─────────────────┘    └─────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│  │ Model Provider  │    │   MCP Tools     │    │  smolagents    ││
│  │   (Auto)        │    │   (Dynamic)     │    │  CodeAgent     ││
│  └─────────────────┘    └─────────────────┘    └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Supported AI Providers

| Provider | Type | Requirements |
|----------|------|--------------|
| OpenAI GPT | API | `OPENAI_API_KEY` |
| Anthropic Claude | API | `ANTHROPIC_API_KEY` |
| Google Gemini | API | `GOOGLE_API_KEY` |
| Ollama | Local | Ollama installed with models |
| LM Studio | Local | LM Studio running local server |

## 🛠️ Domain Capabilities

### Coding Domain
- Execute Python code safely
- Analyze code for bugs and complexity
- Debug and fix code issues
- Generate code based on requirements

### Mathematics Domain
- Solve equations symbolically
- Plot mathematical functions
- Perform complex calculations
- Statistical analysis

### Science Domain
- Unit conversions
- Physics calculations
- Chemistry computations
- Scientific data analysis

### Web Domain
- Search for information
- Fetch and analyze web content
- Real-time data retrieval

## 🎨 Customization

The agent automatically generates custom tools based on your specific needs. It analyzes your prompt and creates specialized functions when existing tools aren't sufficient.

## 🎯 CLI Commands

The beautiful CLI supports these interactive commands:

- **`help`** - Show comprehensive help with examples and usage guide
- **`history`** - View your session history with status indicators
- **`status`** - Display agent status and provider information  
- **`clear`** - Clear the screen and refresh the interface
- **`quit` / `exit` / `q`** - Exit with a beautiful goodbye message

## 🎨 CLI Features

### Beautiful Interface Elements
- **🎨 Rich Colors**: Syntax highlighting and color-coded responses
- **📊 Progress Bars**: Real-time progress for tool generation and AI processing
- **🗂️ Elegant Panels**: Clean, bordered sections for different types of content
- **📈 Status Tables**: Organized display of agent status and session information
- **🔄 Live Updates**: Dynamic progress indicators and loading animations

### Smart Interactions
- **💡 Auto-completion**: Rich prompt with validation
- **📝 Request Tracking**: Keep track of all requests and their success status
- **⚡ Fast Processing**: Optimized for quick response times
- **🎯 Error Handling**: Beautiful error displays with helpful suggestions

## 🔒 Security

- Code execution is handled safely with error catching
- No sensitive information is logged
- Local model support for privacy-sensitive tasks

## 📝 License

MIT License - Feel free to modify and use as needed.

## 🚀 Advanced Usage

You can also use the agent programmatically:

```python
from adaptive_agent import AdaptiveAgent

# Initialize with specific provider
agent = AdaptiveAgent(provider_type="openai", model="gpt-4")

# Run a task
result = agent.run("Calculate the fibonacci sequence up to 100")
print(result)
```

## 🤝 Contributing

This is a single-file implementation designed for easy customization. Feel free to:
- Add new domain templates
- Improve tool generation logic
- Add support for additional AI providers
- Enhance error handling

## 🐛 Troubleshooting

- **No providers available**: Ensure at least one API key is set or local model is running
- **Import errors**: Dependencies will auto-install on first run
- **Tool generation fails**: Check your API key and internet connection
- **Local models not working**: Verify Ollama/LM Studio is running and accessible

---

*Built with smolagents, MCP, and multiple AI providers for maximum adaptability* 🚀