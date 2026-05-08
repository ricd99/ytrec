#!/usr/bin/env python3
"""
Runs sequentially: collect → ETL → train → evaluate
"""
import os
import sys
import time
import argparse
import mlflow
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split

# Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_pipeline.collect_channels import collect
from scripts.data_pipeline.etl import run_etl
from src.db.connection import db_manager
from src.core.config import settings
from src.core.pipeline_state import PipelineState
from src.core.failure_tracker import FailureTracker
from src.features.build_features import build_features
from src.utils.validate_data import validate_data
from src.embedding import batch_encode
from src.models.train import train_model
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model
from huggingface_hub import HfApi
from pathlib import Path


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.
    """
    project_root = Path(__file__).resolve().parent.parent
    pipeline_start_time = time.time()

    # Use S3 for MLflow tracking (persistent across GitHub Actions runs)
    # mlflow_s3_uri = f"s3://{settings.s3_bucket}/mlflow/"
    # mlflow.set_tracking_uri(mlflow_s3_uri)
    # mlflow.set_experiment(args.experiment)

    # with mlflow.start_run():
    pipeline_state = PipelineState()
    failure_tracker = FailureTracker()
    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    pipeline_state.start_run(run_id)

    try:
        # === STAGE 1: Collect New Channels ===
        print("collecting new channels from yt api")
        stage_start = time.time()
        collect()
        # mlflow.log_metric("stage_collect_time", time.time() - stage_start)
        pipeline_state.complete_stage("collect")

        # === STAGE 2: ETL ===
        print("running etl")
        stage_start = time.time()
        new_channel_count = run_etl()
        # mlflow.log_metric("stage_etl_time", time.time() - stage_start)
        pipeline_state.set_new_channel_count(new_channel_count)
        pipeline_state.complete_stage("etl", metadata={"new_channel_count": new_channel_count})

        if new_channel_count == 0:
            print("no new channels found. skipping model training")
            # mlflow.log_metric("pipeline_total_time", time.time() - pipeline_start_time)
            pipeline_state.reset()
            return

        # === STAGE 3: Data Loading ===
        print("Loading data from RDS...")
        stage_start = time.time()
        df_enc = db_manager.fetch_dataframe("SELECT * FROM channels_final")
        # mlflow.log_metric("stage_load_time", time.time() - stage_start)
        print(f"Data loaded: {df_enc.shape[0]} rows, {df_enc.shape[1]} columns")
        pipeline_state.complete_stage("load")

        # Save Feature Metadata for Serving Consistency
        artifacts_dir = os.path.join(project_root, "artifacts", run_id)
        os.makedirs(artifacts_dir, exist_ok=True)

        # Save run_id to latest_run.txt for local development
        latest_run_path = os.path.join(project_root, "artifacts", "latest_run.txt")
        with open(latest_run_path, "w") as f:
            f.write(run_id)

        feature_cols = list(df_enc.columns)

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f)

        # mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.json")
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 3.1: Data Validation ===
        print("Validating data quality...")
        stage_start = time.time()
        is_valid, failed = validate_data(df_enc)
        # mlflow.log_metric("stage_validate_time", time.time() - stage_start)
        # mlflow.log_metric("data_quality_pass", int(is_valid))
        pipeline_state.record_validation(is_valid, failed)
        pipeline_state.complete_stage("validate")

        # Data drift detection
        failure_rate = pipeline_state.get_validation_failure_rate(last_n=10)
        if failure_rate > 0.2:
            print(f"WARNING: Data drift detected! Validation failure rate: {failure_rate:.1%}")
            # mlflow.log_metric("data_drift_warning", 1)

        if not is_valid:
            # mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            pipeline_state.reset()
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        # === STAGE 4: Model Training (Tuning + Training) ===
        print("Training Nearest Neighbors model...")

        df_train, df_test = train_test_split(df_enc, train_size=0.98, random_state=67)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        print(f"Train: {df_train.shape[0]} samples | Test: {df_test.shape[0]} samples")

        # Get hyperparameters (currently returns hardcoded values - could be enhanced with Optuna)
        stage_start = time.time()
        best_params = tune_model(df_train, df_test)
        # mlflow.log_metric("stage_tune_time", time.time() - stage_start)
        # mlflow.log_params(best_params)
        print(f"Hyperparameters: {best_params}")
        pipeline_state.complete_stage("train", metadata={"best_params": best_params})

        # === STAGE 5: Final Training ===
        print("Training optimized Nearest Neighbors model...")
        stage_start = time.time()
        nn, embeddings, df_lookup = train_model(df_train, params=best_params)
        train_time = time.time() - stage_start
        # mlflow.log_metric("train_time", train_time)
        # mlflow.log_metric("stage_train_time", train_time)
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Evaluating===
        print("Evaluating model...")
        stage_start = time.time()
        test_texts = df_test["text"].fillna("").tolist()
        test_embeddings = batch_encode(test_texts)
        mean_dist, median_dist = evaluate_model(nn, test_embeddings)
        # mlflow.log_metric("stage_evaluate_time", time.time() - stage_start)
        # mlflow.log_metric("mean_nn_distance", mean_dist)
        # mlflow.log_metric("median_nn_distance", median_dist)
        print(f"Mean distance: {mean_dist}")
        print(f"Median distance: {median_dist}")
        pipeline_state.complete_stage("evaluate", metadata={"mean_dist": mean_dist, "median_dist": median_dist})

        # === STAGE 7: Save Model===
        print("Saving model...")
        stage_start = time.time()
        joblib.dump(nn, os.path.join(artifacts_dir, "nn_model.pkl"))
        joblib.dump(embeddings, os.path.join(artifacts_dir, "embeddings.pkl"))
        joblib.dump(df_lookup, os.path.join(artifacts_dir, "df_lookup.pkl"))

        # mlflow.log_artifact(os.path.join(artifacts_dir, "nn_model.pkl"))
        # mlflow.log_artifact(os.path.join(artifacts_dir, "embeddings.pkl"))
        # mlflow.log_artifact(os.path.join(artifacts_dir, "df_lookup.pkl"))
        # mlflow.log_metric("stage_save_time", time.time() - stage_start)
        print(f"Model and embeddings and lookup table saved to {artifacts_dir}")
        pipeline_state.complete_stage("save", metadata={"artifacts_dir": artifacts_dir})

        # === STAGE 8: Upload to Hugging Face Hub ===
        print("Uploading artifacts to Hugging Face Hub...")
        stage_start = time.time()
        
        api = HfApi()

        # Upload each artifact
        artifacts_to_upload = ["nn_model.pkl", "embeddings.pkl", "df_lookup.pkl", "feature_columns.json"]
        for filename in artifacts_to_upload:
            filepath = os.path.join(artifacts_dir, filename)
            if os.path.exists(filepath):
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=settings.hf_repo_id,
                )
                print(f"Uploaded {filename} to {settings.hf_repo_id}")

        # mlflow.log_metric("stage_upload_time", time.time() - stage_start)
        print("Artifacts uploaded to Hugging Face Hub")
        pipeline_state.complete_stage("upload")

        # Log total pipeline time
        total_time = time.time() - pipeline_start_time
        # mlflow.log_metric("pipeline_total_time", total_time)
        print(f"Pipeline completed in {total_time:.2f} seconds")
        pipeline_state.reset()

    except Exception as e:
        stage = pipeline_state.state.get("last_successful_stage", "collect")
        error_msg = str(e)
        print(f"Pipeline failed at stage: {stage}")
        print(f"Error: {error_msg}")

        # Log failure to S3 (not GitHub Issues)
        failure_tracker.log_failure(
            stage=stage,
            error=error_msg,
            run_id=run_id,
            metadata={"new_channel_count": pipeline_state.state.get("new_channel_count", 0)}
        )

        # Log to MLflow
        # mlflow.log_metric("pipeline_failed", 1)
        # mlflow.log_param("failure_stage", stage)
        # mlflow.log_param("failure_error", error_msg[:500])  # Truncate long errors

        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run youtube pipeline with NearestNeighbors + Sentence Embeddings + MLflow")
    p.add_argument("--experiment", type=str, default="Youtube Recommender")
    # p.add_argument("--mlflow_uri", type=str, default=None)

    args = p.parse_args()
    main(args)
