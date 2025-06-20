# Architecture Overview

## Core Components

### 1. AdaptiveAgent
The main agent class that coordinates everything:
- Manages AI providers
- Orchestrates tool generation
- Handles user interactions

### 2. MCPToolGenerator
Dynamically generates tools using AI:
- Analyzes user prompts
- Creates custom Python functions
- Applies `@tool` decorators
- Executes generated code safely

### 3. ModelProviders
Support for multiple AI providers:
- **OpenAIProvider**: GPT-3.5, GPT-4, etc.
- **AnthropicProvider**: Claude models
- **GeminiProvider**: Google's Gemini
- **OllamaProvider**: Local models via Ollama
- **LMStudioProvider**: Local LM Studio

### 4. CLI Interface
Beautiful terminal interface:
- Rich colors and animations
- Progress bars and panels
- Session history tracking
- Interactive commands

## Data Flow

```
User Input
    ↓
Prompt Analysis (AI)
    ↓
Tool Generation (AI)
    ↓
Tool Execution
    ↓
smolagents CodeAgent
    ↓
Final Result
```

## Tool Generation Process

1. **Analysis**: AI analyzes the user request
2. **Planning**: Determines what tools are needed
3. **Generation**: AI writes Python functions with proper decorators
4. **Validation**: Code is cleaned and validated
5. **Execution**: Functions are compiled and added to agent
6. **Usage**: smolagents uses tools to complete the task

## Security

- No pre-built tools (everything generated fresh)
- Code execution in controlled namespace
- Input validation and sanitization
- Error handling and graceful failures

## Extensibility

- Easy to add new AI providers
- Modular tool generation system
- Plugin-like architecture
- Configurable behavior