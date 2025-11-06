#!/bin/bash

# Base directory
BASE_DIR="."

# Function to create directory and file
create_dir_and_file() {
    local dir=$1
    local file=$2
    local created=0
    
    # Create directory if it doesn't exist
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        created=1
    fi
    
    # Create file if it doesn't exist and it's specified
    if [ ! -z "$file" ] && [ ! -f "$dir/$file" ]; then
        touch "$dir/$file"
        # Add __init__.py content if it's a Python file
        if [[ "$file" == "__init__.py" ]]; then
            echo "# Initialize $dir module" > "$dir/$file"
        fi
        
        # Only print if file was actually created
        echo "Created: $dir/$file"
    fi
}

# Create all directories and files
# Components
create_dir_and_file "$BASE_DIR/components/screens" "__init__.py"
create_dir_and_file "$BASE_DIR/components/screens" "preprocessing_screen.py"
create_dir_and_file "$BASE_DIR/components/screens" "traditional_screen.py"
create_dir_and_file "$BASE_DIR/components/screens" "deep_learning_screen.py"
create_dir_and_file "$BASE_DIR/components/screens" "gnn_screen.py"
create_dir_and_file "$BASE_DIR/components/screens" "comparison_screen.py"
create_dir_and_file "$BASE_DIR/components/screens" "integration_screen.py"

# Data preprocessing
create_dir_and_file "$BASE_DIR/data_preprocessing" "__init__.py"
create_dir_and_file "$BASE_DIR/data_preprocessing" "preprocessing.py"
create_dir_and_file "$BASE_DIR/data_preprocessing" "feature_engineering.py"
create_dir_and_file "$BASE_DIR/data_preprocessing" "data_augmentation.py"

create_dir_and_file "$BASE_DIR/data_preprocessing/vietnamese_utils" "__init__.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/vietnamese_utils" "word_segmentation.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/vietnamese_utils" "spell_checker.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/vietnamese_utils" "local_dict.py"

create_dir_and_file "$BASE_DIR/data_preprocessing/text_cleaning" "__init__.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/text_cleaning" "html_cleaner.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/text_cleaning" "emoji_handler.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/text_cleaning" "url_remover.py"

create_dir_and_file "$BASE_DIR/data_preprocessing/vectorization" "__init__.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/vectorization" "tfidf_vectorizer.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/vectorization" "word2vec_trainer.py"
create_dir_and_file "$BASE_DIR/data_preprocessing/vectorization" "embedding_processor.py"

# Model training
create_dir_and_file "$BASE_DIR/model_training/base" "__init__.py"
create_dir_and_file "$BASE_DIR/model_training/base" "trainer.py"
create_dir_and_file "$BASE_DIR/model_training/base" "model.py"

create_dir_and_file "$BASE_DIR/model_training/traditional" "__init__.py"
create_dir_and_file "$BASE_DIR/model_training/traditional" "svm_trainer.py"
create_dir_and_file "$BASE_DIR/model_training/traditional" "logistic_trainer.py"
create_dir_and_file "$BASE_DIR/model_training/traditional" "random_forest_trainer.py"

create_dir_and_file "$BASE_DIR/model_training/deep_learning" "__init__.py"
create_dir_and_file "$BASE_DIR/model_training/deep_learning" "phobert_trainer.py"
create_dir_and_file "$BASE_DIR/model_training/deep_learning" "bilstm_trainer.py"
create_dir_and_file "$BASE_DIR/model_training/deep_learning" "embedding_trainer.py"

create_dir_and_file "$BASE_DIR/model_training/graph" "__init__.py"
create_dir_and_file "$BASE_DIR/model_training/graph" "graph_builder.py"
create_dir_and_file "$BASE_DIR/model_training/graph" "gcn_trainer.py"
create_dir_and_file "$BASE_DIR/model_training/graph" "gat_trainer.py"

create_dir_and_file "$BASE_DIR/model_training/optimization" "__init__.py"
create_dir_and_file "$BASE_DIR/model_training/optimization" "grid_search.py"
create_dir_and_file "$BASE_DIR/model_training/optimization" "optuna_optimizer.py"

# Model evaluation
create_dir_and_file "$BASE_DIR/model_evaluation/metrics" "__init__.py"
create_dir_and_file "$BASE_DIR/model_evaluation/metrics" "classification_metrics.py"
create_dir_and_file "$BASE_DIR/model_evaluation/metrics" "ranking_metrics.py"

create_dir_and_file "$BASE_DIR/model_evaluation/visualization" "__init__.py"
create_dir_and_file "$BASE_DIR/model_evaluation/visualization" "confusion_matrix.py"
create_dir_and_file "$BASE_DIR/model_evaluation/visualization" "learning_curves.py"
create_dir_and_file "$BASE_DIR/model_evaluation/visualization" "feature_importance.py"

create_dir_and_file "$BASE_DIR/model_evaluation/comparison" "__init__.py"
create_dir_and_file "$BASE_DIR/model_evaluation/comparison" "model_comparator.py"
create_dir_and_file "$BASE_DIR/model_evaluation/comparison" "statistical_tests.py"

create_dir_and_file "$BASE_DIR/model_evaluation/reports" "__init__.py"
create_dir_and_file "$BASE_DIR/model_evaluation/reports" "performance_report.py"
create_dir_and_file "$BASE_DIR/model_evaluation/reports" "error_analysis.py"

# Pipelines
create_dir_and_file "$BASE_DIR/pipelines/base" "__init__.py"
create_dir_and_file "$BASE_DIR/pipelines/base" "pipeline.py"

create_dir_and_file "$BASE_DIR/pipelines/data" "__init__.py"
create_dir_and_file "$BASE_DIR/pipelines/data" "crawler_pipeline.py"
create_dir_and_file "$BASE_DIR/pipelines/data" "preprocessing_pipeline.py"
create_dir_and_file "$BASE_DIR/pipelines/data" "feature_pipeline.py"

create_dir_and_file "$BASE_DIR/pipelines/training" "__init__.py"
create_dir_and_file "$BASE_DIR/pipelines/training" "traditional_pipeline.py"
create_dir_and_file "$BASE_DIR/pipelines/training" "deep_learning_pipeline.py"
create_dir_and_file "$BASE_DIR/pipelines/training" "graph_pipeline.py"

create_dir_and_file "$BASE_DIR/pipelines/evaluation" "__init__.py"
create_dir_and_file "$BASE_DIR/pipelines/evaluation" "metrics_pipeline.py"
create_dir_and_file "$BASE_DIR/pipelines/evaluation" "reporting_pipeline.py"

# Config
create_dir_and_file "$BASE_DIR/config/preprocessing" "__init__.py"
create_dir_and_file "$BASE_DIR/config/preprocessing" "cleaning_config.py"
create_dir_and_file "$BASE_DIR/config/preprocessing" "vectorization_config.py"

create_dir_and_file "$BASE_DIR/config/models" "__init__.py"
create_dir_and_file "$BASE_DIR/config/models" "traditional_config.py"
create_dir_and_file "$BASE_DIR/config/models" "deep_learning_config.py"
create_dir_and_file "$BASE_DIR/config/models" "graph_config.py"

create_dir_and_file "$BASE_DIR/config/training" "__init__.py"
create_dir_and_file "$BASE_DIR/config/training" "hyperparameters.py"
create_dir_and_file "$BASE_DIR/config/training" "optimization_config.py"

create_dir_and_file "$BASE_DIR/config/pipeline" "__init__.py"
create_dir_and_file "$BASE_DIR/config/pipeline" "data_config.py"
create_dir_and_file "$BASE_DIR/config/pipeline" "evaluation_config.py"

# Experiments
create_dir_and_file "$BASE_DIR/experiments/notebooks" "data_analysis.ipynb"
create_dir_and_file "$BASE_DIR/experiments/notebooks" "model_prototyping.ipynb"
create_dir_and_file "$BASE_DIR/experiments/notebooks" "evaluation_analysis.ipynb"

create_dir_and_file "$BASE_DIR/experiments/scripts" "run_experiment.py"
create_dir_and_file "$BASE_DIR/experiments/scripts" "batch_training.py"
create_dir_and_file "$BASE_DIR/experiments/scripts" "results_aggregator.py"

create_dir_and_file "$BASE_DIR/experiments/configs" "experiment_001.yaml"
create_dir_and_file "$BASE_DIR/experiments/configs" "experiment_002.yaml"

# Deployment
create_dir_and_file "$BASE_DIR/deployment/api" "__init__.py"
create_dir_and_file "$BASE_DIR/deployment/api" "routes.py"
create_dir_and_file "$BASE_DIR/deployment/api" "handlers.py"

create_dir_and_file "$BASE_DIR/deployment/serving" "__init__.py"
create_dir_and_file "$BASE_DIR/deployment/serving" "model_server.py"
create_dir_and_file "$BASE_DIR/deployment/serving" "batch_processor.py"

create_dir_and_file "$BASE_DIR/deployment/monitoring" "__init__.py"
create_dir_and_file "$BASE_DIR/deployment/monitoring" "metrics_collector.py"
create_dir_and_file "$BASE_DIR/deployment/monitoring" "alert_system.py"

# Utils
create_dir_and_file "$BASE_DIR/utils/ml" "__init__.py"
create_dir_and_file "$BASE_DIR/utils/ml" "data_utils.py"
create_dir_and_file "$BASE_DIR/utils/ml" "training_utils.py"
create_dir_and_file "$BASE_DIR/utils/ml" "visualization_utils.py"

# Tests
create_dir_and_file "$BASE_DIR/tests/test_preprocessing" "__init__.py"
create_dir_and_file "$BASE_DIR/tests/test_models" "__init__.py"
create_dir_and_file "$BASE_DIR/tests/test_evaluation" "__init__.py"
create_dir_and_file "$BASE_DIR/tests/test_pipeline" "__init__.py"

# Docs
create_dir_and_file "$BASE_DIR/docs" "preprocessing.md"
create_dir_and_file "$BASE_DIR/docs" "models.md"
create_dir_and_file "$BASE_DIR/docs" "evaluation.md"
create_dir_and_file "$BASE_DIR/docs" "api.md"