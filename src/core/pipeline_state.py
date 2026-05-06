"""
Pipeline state management for checkpointing and resuming.
Tracks which stages have completed and handles error recovery.
"""
import json
import boto3
from pathlib import Path
from datetime import datetime
from src.core.config import settings
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATE_FILE = PROJECT_ROOT / "data" / "misc" / "pipeline_state.json"


class PipelineState:
    """Manages pipeline execution state for checkpointing and resuming."""
    
    STAGES = ["collect", "etl", "load", "validate", "train", "evaluate", "save", "upload"]
    
    def __init__(self):
        self.s3 = boto3.client("s3", region_name="us-west-2")
        self.bucket = settings.s3_bucket
        self.state_file = STATE_FILE
        self.state = self._load()
    
    def _load(self):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key="pipeline_monitoring/pipeline_state.json")
            return json.loads(response["Body"].read())
        except:
            return self._default_state()
    
    def _default_state(self) -> dict:
        """Return default (empty) state."""
        return {
            "last_run": None,
            "last_successful_stage": None,
            "stages_completed": [],
            "new_channel_count": 0,
            "validation_history": [],
            "run_id": None
        }
    
    def _save(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key="pipeline_monitoring/pipeline_state.json",
            Body=json.dumps(self.state, indent=2)
        )
    
    def start_run(self, run_id: str):
        """Mark the start of a pipeline run."""
        self.state["run_id"] = run_id
        self.state["last_run"] = datetime.now().isoformat()
        self.state["stages_completed"] = []
        self._save()
    
    def complete_stage(self, stage: str, metadata: Optional[dict] = None):
        """Mark a stage as completed."""
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        if stage not in self.state["stages_completed"]:
            self.state["stages_completed"].append(stage)
        self.state["last_successful_stage"] = stage
        if metadata:
            self.state[f"{stage}_metadata"] = metadata
        self._save()
    
    def is_stage_complete(self, stage: str) -> bool:
        """Check if a stage was completed in the current run."""
        return stage in self.state["stages_completed"]
    
    def should_skip_stage(self, stage: str) -> bool:
        """Determine if a stage should be skipped (completed in current run)."""             #TODO: not implemented (should_skip_stage and is_stage_complete)
        return self.is_stage_complete(stage)
    
    def set_new_channel_count(self, count: int):
        """Record the number of new channels processed."""
        self.state["new_channel_count"] = count
        self._save()
    
    def record_validation(self, is_valid: bool, failed: list):
        """Record data validation results for drift detection."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "is_valid": is_valid,
            "failed_count": len(failed),
            "run_id": self.state.get("run_id")
        }
        self.state["validation_history"].append(entry)
        # Keep only last 30 entries
        self.state["validation_history"] = self.state["validation_history"][-30:]
        self._save()
    
    def get_validation_failure_rate(self, last_n: int = 10) -> float:
        """Calculate validation failure rate over last n runs."""
        history = self.state["validation_history"][-last_n:]
        if not history:
            return 0.0
        failures = sum(1 for h in history if not h["is_valid"])
        return failures / len(history)
    
    def reset(self):
        """Reset state (for fresh start)."""
        self.state = self._default_state()
        self._save()
