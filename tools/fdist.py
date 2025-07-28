import os
import sys
import argparse
from collections import defaultdict

try:
    from tqdm import tqdm
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("\nMissing required packages: [rich, tqdm]")
    print("Install them using:\n\n    pip install rich tqdm\n")
    sys.exit(1)

def banner():
    console = Console()
    console.print("\n[bold cyan]fdist[/] - [green]File Extension Distribution Tool[/]\n", justify="center")

def collect(path):
    files = []
    for root, _, filenames in os.walk(path):
        for name in filenames:
            files.append(os.path.join(root, name))

    stats = defaultdict(int)
    bar = tqdm(files, bar_format="{percentage:3.0f}%|{bar}|", ncols=80)
    for file in bar:
        ext = os.path.splitext(file)[1].lower() or 'no_extension'
        stats[ext] += 1

    return stats, len(files)

def render(stats, total):
    console = Console()
    console.print(f"\n[bold green]Total files detected:[/] [yellow]{total}[/]\n")

    table = Table(title="Extension Distribution", style="cyan")
    table.add_column("Extension", style="magenta")
    table.add_column("Percentage", style="green", justify="right")

    for ext, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        percent = (count / total) * 100 if total else 0
        table.add_row(ext, f"{percent:.2f}%")

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="fdist - File Extension Distribution Tool")
    parser.add_argument("path", help="Target folder path")
    args = parser.parse_args(sys.argv[1:] or ["-h"])
    banner()
    stats, total = collect(args.path)
    render(stats, total)

if __name__ == "__main__":
    main()
