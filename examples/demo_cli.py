#!/usr/bin/env python3
"""
Demo of the beautiful CLI interface
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_agent import AdaptiveAgent
from rich.console import Console
from rich.panel import Panel

def demo_cli():
    """Demonstrate the CLI capabilities"""
    console = Console()
    
    # Show header
    console.clear()
    header = Panel(
        "[bold cyan]🤖 ADAPTIVE AGENT CLI DEMO[/bold cyan]\n"
        "[italic bright_blue]Beautiful, Interactive Interface[/italic bright_blue]",
        border_style="bright_cyan",
        padding=(1, 2)
    )
    console.print(header)
    
    # Initialize agent
    console.print("\n[cyan]Initializing agent...[/cyan]")
    try:
        agent = AdaptiveAgent(provider_type="auto")
        console.print("[green]✅ Agent ready![/green]")
        
        # Demo requests
        test_cases = [
            "solve 3 + 4",
            "calculate sqrt(16)",
            "help",
        ]
        
        for i, request in enumerate(test_cases, 1):
            console.print(f"\n[bold yellow]Demo {i}: {request}[/bold yellow]")
            
            # Show input panel
            input_panel = Panel(
                f"[bold cyan]💭 {request}[/bold cyan]",
                title="[bold]Request[/bold]",
                border_style="cyan"
            )
            console.print(input_panel)
            
            if request == "help":
                # Show help content
                help_panel = Panel(
                    "[green]🎉 The Adaptive Agent can handle any request!\n"
                    "✨ It generates custom tools dynamically using real AI.[/green]",
                    title="[bold bright_green]🚀 Help[/bold bright_green]",
                    border_style="green"
                )
                console.print(help_panel)
            else:
                try:
                    result = agent.run(request, silent=True)
                    result_panel = Panel(
                        f"[bold green]{result}[/bold green]",
                        title="[bold bright_green]🎯 Result[/bold bright_green]",
                        border_style="green"
                    )
                    console.print(result_panel)
                except Exception as e:
                    error_panel = Panel(
                        f"[red]Error: {str(e)}[/red]",
                        title="[bold red]❌ Error[/bold red]",
                        border_style="red"
                    )
                    console.print(error_panel)
        
        # Final message
        final_panel = Panel(
            "[bold bright_yellow]🎉 CLI Demo Complete!\n"
            "🚀 Run 'python3 adaptive_agent.py' for full interactive experience![/bold bright_yellow]",
            title="[bold]Demo Complete[/bold]",
            border_style="yellow"
        )
        console.print(final_panel)
        
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {e}[/red]")

if __name__ == "__main__":
    demo_cli()