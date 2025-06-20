# Usage Guide

## Getting Started

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
python run.py
```

### Basic Commands
- `help` - Show help information
- `history` - View session history
- `status` - Show agent status
- `clear` - Clear screen
- `quit` - Exit

## Examples

### Mathematics
```
> solve x^2 + 2x - 3 = 0
> calculate sqrt(16) + 5*3
> plot sin(x) from -10 to 10
```

### Programming
```
> write Python code to find prime numbers
> debug this function: def broken_func(): return x + 1
> analyze code complexity for bubble sort
```

### Science
```
> convert 100 fahrenheit to celsius
> calculate molecular weight of H2O
> what is the speed of light in m/s
```

### Web & Research
```
> search for latest developments in quantum computing
> find information about machine learning algorithms
> lookup current weather in New York
```

## Configuration

### API Keys
Set environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

### Local Models
- **Ollama**: Install Ollama and pull models: `ollama pull llama2`
- **LM Studio**: Install LM Studio and start local server

## Advanced Usage

### Silent Mode
```python
from src.adaptive_agent import AdaptiveAgent

agent = AdaptiveAgent(provider_type="openai")
result = agent.run("solve 2+2", silent=True)
print(result)
```

### Custom Provider
```python
agent = AdaptiveAgent(
    provider_type="anthropic",
    model="claude-3-sonnet-20240229"
)
```