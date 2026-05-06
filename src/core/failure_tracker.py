"""
Failure tracker that logs pipeline failures to S3 for monitoring.
"""
import json
import boto3
from datetime import datetime
from src.core.config import settings


class FailureTracker:
    """Logs pipeline failures to S3 for monitoring without GitHub Issues."""
    
    def __init__(self):
        self.s3 = boto3.client("s3", region_name="us-west-2")
        self.bucket = settings.s3_bucket
        self.key = "pipeline_monitoring/failures.json"
    
    def log_failure(self, stage: str, error: str, run_id: str, metadata: dict = None):
        """Log a failure to S3."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error": str(error),
            "run_id": run_id,
            "metadata": metadata or {}
        }
        
        # Fetch existing failures
        failures = self._fetch_failures()
        failures.append(entry)
        
        # Keep only last 100 failures
        failures = failures[-100:]
        
        # Upload to S3
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key,
            Body=json.dumps(failures, indent=2),
            ContentType="application/json"
        )
        print(f"Failure logged to S3: {self.key}")
    
    def _fetch_failures(self) -> list:
        """Fetch existing failures from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            return json.loads(response["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return []
        except Exception:
            return []
    
    def get_recent_failures(self, count: int = 10) -> list:
        """Get the most recent failures."""
        failures = self._fetch_failures()
        return failures[-count:]
