"""Hugging Face Spaces Entrypoint (100% Free Gradio / Native SDK).

Compatible with:
- Hugging Face Spaces (sdk: gradio, ZeroGPU / CPU, 16GB free RAM)
- Local Python execution (python app.py)
"""

import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import get_harness

# Support Hugging Face ZeroGPU acceleration if enabled in Space settings
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is None:
                return lambda f: f
            return fn


@spaces.GPU(duration=60)
def execute_distributed_training_step():
    """Execute distributed training step with ZeroGPU acceleration on Hugging Face."""
    harness = get_harness()
    return harness.train_step()


def create_gradio_app():
    """Build a Gradio Blocks application wired directly into RMLTP cluster harness."""
    import gradio as gr

    harness = get_harness()

    def refresh_stats():
        status = harness.get_cluster_status()
        active_w = len(status.get("workers", []))
        step = status.get("step", 0)
        loss = status.get("loss", 0.0)
        throughput = status.get("throughput", 0.0)
        logs_str = "\n".join(status.get("logs", []))

        workers_data = []
        for w in status.get("workers", []):
            cpu_pct = f"{w.get('cpu_percent', 0.0)}%"
            last_seen = f"{w.get('last_seen', 0.0)}s ago"
            workers_data.append([
                str(w.get("id", "worker")),
                str(w.get("status", "active")).upper(),
                f"{loss:.4f}",
                cpu_pct,
                last_seen,
                str(w.get("gpu", "N/A")),
            ])

        return (
            f"Active: {active_w}/2",
            str(step),
            f"{loss:.4f}",
            f"{throughput:.1f} samples/s",
            workers_data,
            logs_str,
        )

    def on_train_step():
        execute_distributed_training_step()
        return refresh_stats()

    def on_simulate_crash():
        harness.simulate_crash()
        return refresh_stats()

    def on_submit_job(name, dataset, num_workers):
        harness.submit_job({
            "job_name": name,
            "dataset_location": dataset,
            "num_workers": int(num_workers),
        })
        return refresh_stats()

    with gr.Blocks(title="RMLTP Distributed ML Platform", theme=gr.themes.Soft()) as demo_app:
        gr.Markdown(
            """
            # ⚡ RMLTP — Distributed Machine Learning Platform
            **Kubernetes-style Orchestration + Ray Dynamic Scheduling + PyTorch Distributed Core**
            *Running live in free Hugging Face Space (ZeroMQ TCP transport + ModelRunner)*
            """
        )

        with gr.Row():
            stat_workers = gr.Textbox(label="Cluster Nodes", interactive=False)
            stat_step = gr.Textbox(label="Global Step", interactive=False)
            stat_loss = gr.Textbox(label="Training Loss", interactive=False)
            stat_throughput = gr.Textbox(label="Throughput", interactive=False)

        with gr.Row():
            btn_step = gr.Button("🚀 Trigger Distributed Training Step", variant="primary")
            btn_crash = gr.Button("💥 Simulate Worker Crash / Failover", variant="stop")
            btn_refresh = gr.Button("🔄 Refresh Telemetry", variant="secondary")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 🖥️ Cluster Worker Nodes")
                workers_table = gr.Dataframe(
                    headers=["Node ID", "Status", "Loss", "CPU Usage", "Last Seen", "Hardware"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    interactive=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📜 Real-Time Cluster Event Stream")
                log_box = gr.TextArea(
                    label="Logs",
                    lines=10,
                    interactive=False,
                )

        with gr.Accordion("➕ Submit New Distributed Training Job", open=False):
            with gr.Row():
                in_job_name = gr.Textbox(label="Job Name", value="cifar10-resnet50")
                in_dataset = gr.Textbox(label="Dataset Location", value="s3://ml-datasets/cifar10")
                in_workers = gr.Slider(label="Worker Count", minimum=1, maximum=4, value=2, step=1)
            btn_submit = gr.Button("Submit Job", variant="primary")

        # Wire callbacks
        btn_step.click(on_train_step, outputs=[stat_workers, stat_step, stat_loss, stat_throughput, workers_table, log_box])
        btn_crash.click(on_simulate_crash, outputs=[stat_workers, stat_step, stat_loss, stat_throughput, workers_table, log_box])
        btn_refresh.click(refresh_stats, outputs=[stat_workers, stat_step, stat_loss, stat_throughput, workers_table, log_box])
        btn_submit.click(on_submit_job, inputs=[in_job_name, in_dataset, in_workers], outputs=[stat_workers, stat_step, stat_loss, stat_throughput, workers_table, log_box])

        # Initial load
        demo_app.load(refresh_stats, outputs=[stat_workers, stat_step, stat_loss, stat_throughput, workers_table, log_box])

    return demo_app


# Top-level demo object for Gradio hot-reload and Hugging Face Spaces runner
try:
    import gradio as gr
    demo = create_gradio_app()
except ImportError:
    demo = None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    if demo is not None:
        print(f"\n🚀 Launching RMLTP Gradio Dashboard on 0.0.0.0:{port}...\n")
        demo.launch(server_name="0.0.0.0", server_port=port, share=False)
    else:
        from dashboard import app
        print(f"\n🚀 Gradio not found. Launching RMLTP Flask Dashboard on 0.0.0.0:{port}...\n")
        app.run(host="0.0.0.0", port=port, debug=False)
