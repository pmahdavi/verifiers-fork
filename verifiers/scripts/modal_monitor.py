#!/usr/bin/env python3
"""
Modal Training Monitor TUI

A comprehensive Terminal User Interface for monitoring Modal training runs.
Displays apps, containers, GPU metrics, memory usage, and real-time logs.

Usage:
    vf-modal-monitor
    # or
    uv run vf-modal-monitor

Keyboard Shortcuts:
    q - Quit
    r - Refresh all data
    a - Focus on apps panel
    l - Focus on logs panel
    ↑/↓ - Navigate apps list
    Enter - Select app to view logs
"""

import asyncio
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Optional

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Footer, Static, DataTable, Log
from textual.widget import Widget


class GPUMetrics(Static):
    """Widget to display GPU metrics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_data = []

    async def update_metrics(self, container_id: Optional[str] = None):
        """Fetch and update GPU metrics"""
        if not container_id:
            self.update("No active container selected")
            return

        try:
            # Query GPU metrics from container
            cmd = [
                "modal", "container", "exec", container_id, "--",
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                self.gpu_data = []

                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 8:
                        self.gpu_data.append({
                            'index': parts[0],
                            'name': parts[1],
                            'gpu_util': parts[2],
                            'mem_util': parts[3],
                            'mem_total': parts[4],
                            'mem_used': parts[5],
                            'mem_free': parts[6],
                            'temp': parts[7]
                        })

                self._render_metrics()
            else:
                self.update("Failed to fetch GPU metrics")
        except subprocess.TimeoutExpired:
            self.update("GPU query timeout")
        except Exception as e:
            self.update(f"Error: {str(e)}")

    def _render_metrics(self):
        """Render GPU metrics in a formatted table"""
        if not self.gpu_data:
            self.update("No GPU data available")
            return

        output = "[bold cyan]GPU Metrics[/bold cyan]\n\n"

        for gpu in self.gpu_data:
            # GPU utilization bar
            gpu_util = int(gpu['gpu_util']) if gpu['gpu_util'].isdigit() else 0
            mem_util = int(gpu['mem_util']) if gpu['mem_util'].isdigit() else 0

            gpu_bar = self._create_bar(gpu_util, 20, "green", "red")
            mem_bar = self._create_bar(mem_util, 20, "blue", "yellow")

            output += f"[bold]GPU {gpu['index']}:[/bold] {gpu['name']}\n"
            output += f"  GPU Usage: {gpu_bar} {gpu_util:>3}%\n"
            output += f"  Mem Usage: {mem_bar} {mem_util:>3}%\n"
            output += f"  Memory: {gpu['mem_used']:>6} / {gpu['mem_total']:>6} MiB  "
            output += f"(Free: {gpu['mem_free']:>6} MiB)\n"
            output += f"  Temp: {gpu['temp']:>3}°C\n\n"

        self.update(output)

    def _create_bar(self, value: int, width: int, low_color: str, high_color: str) -> str:
        """Create a colored progress bar"""
        filled = int((value / 100) * width)
        empty = width - filled

        # Color based on value
        if value < 50:
            color = low_color
        elif value < 80:
            color = "yellow"
        else:
            color = high_color

        bar = f"[{color}]{'█' * filled}[/{color}]"
        bar += f"[dim]{'░' * empty}[/dim]"
        return bar


class MemoryMetrics(Static):
    """Widget to display system memory metrics"""

    async def update_metrics(self, container_id: Optional[str] = None):
        """Fetch and update memory metrics"""
        if not container_id:
            self.update("No active container selected")
            return

        try:
            cmd = ["modal", "container", "exec", container_id, "--", "free", "-m"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')

                # Parse memory info
                for line in lines:
                    if line.startswith('Mem:'):
                        parts = line.split()
                        total = int(parts[1])
                        used = int(parts[2])
                        free = int(parts[3])
                        available = int(parts[6]) if len(parts) > 6 else free

                        usage_pct = int((used / total) * 100) if total > 0 else 0

                        bar = self._create_bar(usage_pct, 30)

                        output = "[bold cyan]System Memory[/bold cyan]\n\n"
                        output += f"Usage: {bar} {usage_pct}%\n\n"
                        output += f"Total:     {total:>10,} MB\n"
                        output += f"Used:      {used:>10,} MB\n"
                        output += f"Free:      {free:>10,} MB\n"
                        output += f"Available: {available:>10,} MB\n"

                        self.update(output)
                        return

                self.update("Could not parse memory info")
            else:
                self.update("Failed to fetch memory metrics")
        except Exception as e:
            self.update(f"Error: {str(e)}")

    def _create_bar(self, value: int, width: int) -> str:
        """Create a colored progress bar"""
        filled = int((value / 100) * width)
        empty = width - filled

        if value < 50:
            color = "green"
        elif value < 80:
            color = "yellow"
        else:
            color = "red"

        bar = f"[{color}]{'█' * filled}[/{color}]"
        bar += f"[dim]{'░' * empty}[/dim]"
        return bar


class AppsTable(DataTable):
    """Widget to display Modal apps"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apps_data = []

    async def update_apps(self):
        """Fetch and update apps list"""
        try:
            result = subprocess.run(
                ["modal", "app", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout:
                self.apps_data = json.loads(result.stdout)
                self._render_table()
            else:
                self.apps_data = []
        except Exception as e:
            self.apps_data = []

    def _render_table(self):
        """Render apps in the table"""
        self.clear(columns=True)

        # Add columns
        self.add_column("App ID", width=25)
        self.add_column("Description", width=30)
        self.add_column("State", width=12)
        self.add_column("Tasks", width=8)
        self.add_column("Created", width=20)

        # Add rows
        for app in self.apps_data:
            state = app.get("State", "unknown")

            # Color code state
            if state == "ephemeral":
                state_str = f"[green]{state}[/green]"
            elif state == "stopped":
                state_str = f"[red]{state}[/red]"
            else:
                state_str = f"[yellow]{state}[/yellow]"

            self.add_row(
                app.get("App ID", ""),
                app.get("Description", ""),
                state_str,
                str(app.get("Tasks", "0")),
                app.get("Created at", "")
            )

    def get_selected_app_id(self) -> Optional[str]:
        """Get the currently selected app ID"""
        if self.cursor_row < len(self.apps_data):
            return self.apps_data[self.cursor_row].get("App ID")
        return None


class ContainersInfo(Static):
    """Widget to display running containers"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.containers = []

    async def update_containers(self):
        """Fetch and update containers list"""
        try:
            result = subprocess.run(
                ["modal", "container", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout:
                self.containers = json.loads(result.stdout)
                self._render_containers()
            else:
                self.containers = []
                self.update("No running containers")
        except Exception as e:
            self.update(f"Error: {str(e)}")

    def _render_containers(self):
        """Render containers info"""
        if not self.containers:
            self.update("No running containers")
            return

        output = "[bold cyan]Running Containers[/bold cyan]\n\n"

        for container in self.containers:
            output += f"[bold]Container ID:[/bold] {container.get('Container ID', 'N/A')}\n"
            output += f"[bold]App ID:[/bold] {container.get('App ID', 'N/A')}\n"
            output += f"[bold]App Name:[/bold] {container.get('App Name', 'N/A')}\n"
            output += f"[bold]Start Time:[/bold] {container.get('Start Time', 'N/A')}\n\n"

        self.update(output)

    def get_first_container_id(self) -> Optional[str]:
        """Get the first container ID"""
        if self.containers:
            return self.containers[0].get("Container ID")
        return None


class LogsViewer(Log):
    """Widget to display streaming logs"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_app_id = None
        self.log_process = None

    async def start_streaming(self, app_id: str):
        """Start streaming logs for an app"""
        # Stop previous stream if any
        await self.stop_streaming()

        self.current_app_id = app_id
        self.clear()
        self.write_line(f"[bold cyan]Streaming logs for app: {app_id}[/bold cyan]\n")

        # Start log streaming in background
        asyncio.create_task(self._stream_logs())

    async def _stream_logs(self):
        """Stream logs from Modal app"""
        if not self.current_app_id:
            return

        try:
            cmd = ["modal", "app", "logs", self.current_app_id, "--timestamps"]

            self.log_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Read logs line by line
            for line in self.log_process.stdout:
                if line.strip():
                    self.write_line(line.rstrip())

        except Exception as e:
            self.write_line(f"[red]Error streaming logs: {str(e)}[/red]")

    async def stop_streaming(self):
        """Stop streaming logs"""
        if self.log_process:
            self.log_process.terminate()
            try:
                self.log_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.log_process.kill()
            self.log_process = None


class ModalMonitorApp(App):
    """Main TUI application for Modal monitoring"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 1fr 2fr;
    }

    #apps-container {
        column-span: 2;
        border: solid $accent;
    }

    #gpu-container {
        border: solid $accent;
    }

    #memory-container {
        border: solid $accent;
    }

    #containers-container {
        border: solid $accent;
    }

    #logs-container {
        column-span: 2;
        border: solid $accent;
    }

    DataTable {
        height: 100%;
    }

    Log {
        height: 100%;
    }

    Static {
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("a", "focus_apps", "Apps"),
        ("l", "focus_logs", "Logs"),
    ]

    def compose(self) -> ComposeResult:
        """Create the UI layout"""
        yield Header()

        # Apps section
        with Container(id="apps-container"):
            yield Static("[bold]Modal Apps[/bold]")
            yield AppsTable(id="apps-table", cursor_type="row")

        # GPU metrics section
        with Container(id="gpu-container"):
            yield GPUMetrics(id="gpu-metrics")

        # Memory metrics section
        with Container(id="memory-container"):
            yield MemoryMetrics(id="memory-metrics")

        # Containers section
        with Container(id="containers-container"):
            yield ContainersInfo(id="containers-info")

        # Logs section
        with Container(id="logs-container"):
            yield LogsViewer(id="logs-viewer")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app when mounted"""
        # Set title
        self.title = "Modal Training Monitor"
        self.sub_title = "Real-time monitoring of Modal runs"

        # Initial data load
        await self.refresh_data()

        # Set up auto-refresh every 5 seconds
        self.set_interval(5, self.refresh_data)

    async def refresh_data(self) -> None:
        """Refresh all data"""
        # Update apps table
        apps_table = self.query_one("#apps-table", AppsTable)
        await apps_table.update_apps()

        # Update containers info
        containers_info = self.query_one("#containers-info", ContainersInfo)
        await containers_info.update_containers()

        # Get first container ID for metrics
        container_id = containers_info.get_first_container_id()

        if container_id:
            # Update GPU metrics
            gpu_metrics = self.query_one("#gpu-metrics", GPUMetrics)
            await gpu_metrics.update_metrics(container_id)

            # Update memory metrics
            memory_metrics = self.query_one("#memory-metrics", MemoryMetrics)
            await memory_metrics.update_metrics(container_id)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle app selection"""
        apps_table = self.query_one("#apps-table", AppsTable)
        app_id = apps_table.get_selected_app_id()

        if app_id:
            logs_viewer = self.query_one("#logs-viewer", LogsViewer)
            await logs_viewer.start_streaming(app_id)

    def action_refresh(self) -> None:
        """Refresh all data"""
        asyncio.create_task(self.refresh_data())

    def action_focus_apps(self) -> None:
        """Focus on apps table"""
        self.query_one("#apps-table").focus()

    def action_focus_logs(self) -> None:
        """Focus on logs viewer"""
        self.query_one("#logs-viewer").focus()


def main():
    """Main entry point"""
    app = ModalMonitorApp()
    app.run()


if __name__ == "__main__":
    main()
