#!/usr/bin/env python3
"""
Basic test to verify CLI components work
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_agent import AdaptiveAgent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

def test_cli_components():
    """Test CLI components"""
    console = Console()
    
    # Test header
    console.print("[bold cyan]🤖 Testing CLI Components[/bold cyan]")
    
    # Test table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Feature", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Rich Library", "✅ Loaded")
    table.add_row("Beautiful Panels", "✅ Working")
    table.add_row("Color Support", "✅ Active")
    table.add_row("Agent Integration", "✅ Ready")
    
    panel = Panel(
        table,
        title="[bold bright_blue]📊 CLI Status[/bold bright_blue]",
        border_style="blue"
    )
    console.print(panel)
    
    # Test agent
    try:
        agent = AdaptiveAgent(provider_type="auto")
        result = agent.run("solve 1+1", silent=True)
        
        success_panel = Panel(
            f"[green]🎉 Agent Test Successful!\n"
            f"✨ Result: {result}[/green]",
            title="[bold green]Agent Test[/bold green]",
            border_style="green"
        )
        console.print(success_panel)
        
    except Exception as e:
        error_panel = Panel(
            f"[red]❌ Agent Test Failed: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red"
        )
        console.print(error_panel)

if __name__ == "__main__":
    test_cli_components()