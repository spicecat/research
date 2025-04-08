"""Main script for running Tree Neural Network experiments."""

import time
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import logging
from sklearn.preprocessing import StandardScaler

from data.data_loader import DataLoader
from models.tree_models import TREENN1, TREENN2, TREENN3
from models.forest_models import FONN1, FONN2, FONN3
from config.settings import MODEL_PARAMS

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize wandb
        wandb.init(
            project="tree-neural-networks",
            name="model-comparison",
            config=MODEL_PARAMS
        )
        
        logger.info("Loading dataset...")
        # Load and preprocess data
        X_train, X_test, y_train, y_test = DataLoader.load_boston_data()
        
        # Scale the data
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))
        
        results = []
        
        # Define models to evaluate
        models_to_evaluate = {
            "Tree Models": [
                ("TREENN1", TREENN1),
                ("TREENN2", TREENN2),
                ("TREENN3", TREENN3)
            ],
            "Forest Models": [
                ("FONN1", FONN1),
                ("FONN2", FONN2),
                ("FONN3", FONN3)
            ]
        }
        
        # Ensure dimensions are integers, not numpy arrays or other types
        input_dim = int(X_train_scaled.shape[1])
        hidden_dim = int(MODEL_PARAMS['hidden_dim'])
        output_dim = int(1)
        num_trees = int(MODEL_PARAMS['num_trees'])
        
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Hidden dimension: {hidden_dim}")
        logger.info(f"Output dimension: {output_dim}")
        logger.info(f"Number of trees (for forest models): {num_trees}")
        
        # Evaluate tree and forest models
        for category, model_list in models_to_evaluate.items():
            logger.info(f"\nEvaluating {category}...")
            
            for name, model in model_list:
                try:
                    logger.info(f"\nInitializing {name}...")
                    
                    # Special case for FONN3 - use lower learning rate and adjusted alpha
                    current_lr = MODEL_PARAMS['learning_rate']
                    alpha_value = 0.5  # Default alpha
                    
                    if name == "FONN3":
                        current_lr = 0.001  # Reduced learning rate for stability
                        alpha_value = 0.1   # Reduced tree contribution for FONN3
                        logger.info(f"Using reduced learning rate for FONN3: {current_lr}")
                        logger.info(f"Using reduced alpha for FONN3: {alpha_value}")
                    elif name == "FONN2":
                        # FONN2 doesn't have an alpha parameter in its constructor
                        logger.info(f"FONN2 uses trees in the hidden layer with standard learning rate: {current_lr}")
                    
                    # Debug model class
                    logger.info(f"Model class: {model.__name__}")
                    
                    # Try a simplified initialization approach
                    try:
                        if name == "FONN2":
                            # FONN2 doesn't expect alpha parameter
                            model_instance = model(
                                X_train=X_train_scaled, 
                                y_train=y_train_scaled, 
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                num_trees=num_trees
                            )
                            logger.info("FONN2 initialized without alpha parameter")
                        elif 'FONN' in name:
                            # Initialize other forest models
                            model_instance = model(
                                X_train=X_train_scaled, 
                                y_train=y_train_scaled, 
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                num_trees=num_trees,
                                alpha=alpha_value  # Pass alpha parameter
                            )
                        else:  # TREENN models
                            # Initialize tree model
                            model_instance = model(
                                X_train=X_train_scaled, 
                                y_train=y_train_scaled, 
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim
                            )
                        
                        logger.info(f"{name} model initialized successfully!")
                    except Exception as init_err:
                        logger.error(f"Error initializing {name}: {str(init_err)}")
                        logger.exception("Initialization traceback:")
                        continue
                    
                    # Try training the model
                    try:
                        # For FONN3, use additional gradient clipping for stability
                        gradient_clip = 0.5 if name == "FONN3" else 1.0
                        
                        metrics = train_and_evaluate_model(
                            name=name,
                            model=model_instance,
                            X_train=X_train_scaled,
                            X_test=X_test_scaled,
                            y_train=y_train_scaled,
                            y_test=y_test_scaled,
                            learning_rate=current_lr,
                            gradient_clip=gradient_clip
                        )
                        
                        if metrics:
                            results.append(metrics)
                            logger.info(f"{name} evaluation successful!")
                    except Exception as train_err:
                        logger.error(f"Error training {name}: {str(train_err)}")
                        logger.exception("Training traceback:")
                        continue
                        
                except Exception as e:
                    logger.error(f"General error during {name} evaluation: {str(e)}")
                    logger.exception("Full traceback:")
                    continue
        
        # Evaluate baseline MLP
        logger.info("\nEvaluating baseline MLP...")
        try:
            mlp_metrics = evaluate_mlp(
                X_train_scaled, X_test_scaled,
                y_train_scaled, y_test_scaled
            )
            if mlp_metrics:
                results.append(mlp_metrics)
        except Exception as e:
            logger.error(f"Error during PureMLP evaluation: {str(e)}")
            logger.exception("PureMLP traceback:")
        
        if results:
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            logger.info("\nResults summary:")
            print(results_df)
            
            # Log final results to wandb
            wandb.log({"final_results": wandb.Table(dataframe=results_df)})
        else:
            logger.error("No results to save!")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Main traceback:")
        raise
    finally:
        wandb.finish()

def validate_model_architecture(model, name):
    """Validate model architecture matches expectations."""
    logger.info(f"\n{name} Architecture:")
    if hasattr(model, 'mlp'):
        logger.info(f"MLP Architecture: {model.mlp}")
    if hasattr(model, 'trees'):
        logger.info(f"Number of trees: {len(model.trees)}")
    if hasattr(model, 'tree'):
        logger.info(f"Using single tree")
    logger.info(f"Input dimension: {model.input_dim}")
    logger.info(f"Hidden dimension: {model.hidden_dim}")
    logger.info(f"Output dimension: {model.output_dim}")

def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, learning_rate, gradient_clip=1.0):
    """Train and evaluate model, separated for better error handling"""
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    # Log model architecture
    logger.info(f"{name} architecture:")
    if hasattr(model, 'input_dim'):
        logger.info(f"Input dimension: {model.input_dim}")
    if hasattr(model, 'hidden_dim'):
        logger.info(f"Hidden dimension: {model.hidden_dim}")
    if hasattr(model, 'output_dim'):
        logger.info(f"Output dimension: {model.output_dim}")
    if hasattr(model, 'alpha'):
        logger.info(f"Alpha (tree contribution): {model.alpha}")
    
    # Log regularization settings
    logger.info(f"Applying regularization: L2 strength={MODEL_PARAMS['l2_strength']}, Dropout rate={MODEL_PARAMS['dropout_rate']}")
    
    # Training loop
    logger.info(f"Training {name}...")
    for epoch in range(MODEL_PARAMS['epochs']):
        # Apply dropout to input data during training (if we have enough features)
        if MODEL_PARAMS['dropout_rate'] > 0 and X_train.shape[1] > 5:
            # Create dropout mask - randomly set some features to 0
            dropout_mask = np.random.binomial(1, 1-MODEL_PARAMS['dropout_rate'], size=X_train.shape)
            # Ensure at least 3 features are active per sample
            for i in range(dropout_mask.shape[0]):
                if np.sum(dropout_mask[i]) < 3:
                    random_indices = np.random.choice(dropout_mask.shape[1], 3, replace=False)
                    dropout_mask[i, random_indices] = 1
            
            # Scale the remaining features to maintain expected value
            dropout_mask = dropout_mask / (1-MODEL_PARAMS['dropout_rate'])
            X_train_dropout = X_train * dropout_mask
        else:
            X_train_dropout = X_train
        
        # Training step with gradient clipping and regularization
        if hasattr(model, 'train_with_reg'):
            # If the model has a specific method for training with regularization
            model.train_with_reg(
                X_train_dropout, y_train, 
                epochs=1, 
                learning_rate=learning_rate, 
                gradient_clip=gradient_clip,
                l2_strength=MODEL_PARAMS['l2_strength']
            )
        else:
            # Apply our own regularization during regular training
            model.train(X_train_dropout, y_train, epochs=1, learning_rate=learning_rate, gradient_clip=gradient_clip)
            
            # Apply L2 regularization manually to model weights if possible
            if MODEL_PARAMS['l2_strength'] > 0:
                if hasattr(model, 'weights_hidden'):
                    model.weights_hidden -= learning_rate * MODEL_PARAMS['l2_strength'] * model.weights_hidden
                if hasattr(model, 'weights_output'):
                    model.weights_output -= learning_rate * MODEL_PARAMS['l2_strength'] * model.weights_output
                
                # We can't directly modify feature_importances_ as it's a read-only property
                # We'll add a note about tree regularization instead
                if MODEL_PARAMS['regularize_trees']:
                    if any(hasattr(model, attr) for attr in ['trees_output', 'trees_hidden', 'trees_input']):
                        logger.info(f"Skipping tree regularization for {name} - feature_importances_ is read-only")
                        # For FONN models, apply additional weight decay to compensate
                        if hasattr(model, 'weights_hidden'):
                            model.weights_hidden -= learning_rate * MODEL_PARAMS['weight_decay'] * model.weights_hidden
                        if hasattr(model, 'weights_output'):
                            model.weights_output -= learning_rate * MODEL_PARAMS['weight_decay'] * model.weights_output
        
        # Every 100 epochs, compute and log metrics
        if epoch % 100 == 0 or epoch == MODEL_PARAMS['epochs'] - 1:
            # Use the regular training data for evaluation
            train_preds = model.forward(X_train)
            test_preds = model.forward(X_test)
            
            # Calculate metrics
            current_loss = mean_squared_error(y_test, test_preds)
            current_r2 = r2_score(y_test, test_preds)
            current_mae = mean_absolute_error(y_test, test_preds)
            
            # Calculate training metrics
            train_mse = mean_squared_error(y_train, train_preds)
            train_r2 = r2_score(y_train, train_preds)
            train_mae = mean_absolute_error(y_train, train_preds)
            
            # Calculate overfitting metrics
            overfit_ratio = train_r2 / (current_r2 + 1e-10)
            
            logger.info(f"{name} Epoch {epoch}:")
            logger.info(f"Train MSE: {train_mse:.4f}, Test MSE: {current_loss:.4f}")
            logger.info(f"Train R²: {train_r2:.4f}, Test R²: {current_r2:.4f}")
            logger.info(f"Train MAE: {train_mae:.4f}, Test MAE: {current_mae:.4f}")
            logger.info(f"Overfit ratio (train R² / test R²): {overfit_ratio:.4f}")
            
            # Log to wandb
            wandb.log({
                f"{name}/train_mae": train_mae,
                f"{name}/train_mse": train_mse,
                f"{name}/train_r2": train_r2,
                f"{name}/test_mae": current_mae,
                f"{name}/test_mse": current_loss,
                f"{name}/test_r2": current_r2,
                f"{name}/overfit_ratio": overfit_ratio,
                "epoch": epoch
            })
    
    # Final evaluation
    train_preds = model.forward(X_train)
    test_preds = model.forward(X_test)
    
    current_loss = mean_squared_error(y_test, test_preds)
    current_r2 = r2_score(y_test, test_preds)
    current_mae = mean_absolute_error(y_test, test_preds)
    
    # Calculate training metrics
    train_mse = mean_squared_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)
    
    # Calculate overfitting metric
    overfit_ratio = train_r2 / (current_r2 + 1e-10) if current_r2 > 0 else float('inf')
    
    end_time = time.time()
    
    metrics = {
        "Model": name,
        "R² Score": current_r2,
        "MAE": current_mae,
        "MSE": current_loss,
        "Time (s)": end_time - start_time,
        "Epochs": MODEL_PARAMS['epochs'],
        "Final LR": learning_rate,
        "Train R²": train_r2,
        "Overfit Ratio": overfit_ratio
    }
    
    logger.info(f"\n{name} Final Metrics:")
    logger.info(f"R² Score: {metrics['R² Score']:.4f}")
    logger.info(f"MAE: {metrics['MAE']:.4f}")
    logger.info(f"MSE: {metrics['MSE']:.4f}")
    logger.info(f"Train R²: {metrics['Train R²']:.4f}")
    logger.info(f"Overfit Ratio: {metrics['Overfit Ratio']:.4f}")
    
    return metrics

def evaluate_mlp(X_train, X_test, y_train, y_test):
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        
        logger.info("Starting PureMLP training...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(MODEL_PARAMS['hidden_dim'], MODEL_PARAMS['hidden_dim']),
            activation='relu',
            solver='adam',
            learning_rate='constant',
            learning_rate_init=MODEL_PARAMS['learning_rate'],
            max_iter=MODEL_PARAMS['epochs'],
            random_state=MODEL_PARAMS['random_state'],
            early_stopping=False,  # Disable early stopping to ensure it runs for exactly 1000 epochs
            batch_size=MODEL_PARAMS['batch_size'],
            momentum=0.9,
            validation_fraction=0.1,
            verbose=True,
            alpha=MODEL_PARAMS['l2_strength']  # Add L2 regularization
        )
        
        logger.info(f"PureMLP with L2 regularization (alpha={MODEL_PARAMS['l2_strength']})")
        
        # Apply dropout to the training data
        if MODEL_PARAMS['dropout_rate'] > 0:
            logger.info(f"Applying dropout with rate {MODEL_PARAMS['dropout_rate']} to MLP input")
            X_train_dropout = np.copy(X_train)
            for i in range(X_train.shape[0]):
                # Create dropout mask
                dropout_mask = np.random.binomial(1, 1-MODEL_PARAMS['dropout_rate'], size=X_train.shape[1])
                # Ensure at least 3 features are active
                if np.sum(dropout_mask) < 3:
                    random_indices = np.random.choice(X_train.shape[1], 3, replace=False)
                    dropout_mask[random_indices] = 1
                # Scale the remaining features
                scale = 1.0 / (1.0 - MODEL_PARAMS['dropout_rate'])
                X_train_dropout[i] = X_train[i] * dropout_mask * scale
        else:
            X_train_dropout = X_train
        
        logger.info("Fitting PureMLP...")
        mlp.fit(X_train_dropout, y_train.ravel())
        
        # Calculate metrics
        train_preds = mlp.predict(X_train).reshape(-1, 1)
        test_preds = mlp.predict(X_test).reshape(-1, 1)
        
        # Test metrics
        r2 = r2_score(y_test, test_preds)
        mae = mean_absolute_error(y_test, test_preds)
        mse = mean_squared_error(y_test, test_preds)
        
        # Training metrics
        train_r2 = r2_score(y_train, train_preds)
        train_mae = mean_absolute_error(y_train, train_preds)
        train_mse = mean_squared_error(y_train, train_preds)
        
        # Calculate overfitting metric
        overfit_ratio = train_r2 / (r2 + 1e-10) if r2 > 0 else float('inf')
        
        # Log to wandb
        wandb.log({
            "PureMLP/train_mae": train_mae,
            "PureMLP/train_mse": train_mse,
            "PureMLP/train_r2": train_r2,
            "PureMLP/test_mae": mae,
            "PureMLP/test_mse": mse,
            "PureMLP/test_r2": r2,
            "PureMLP/overfit_ratio": overfit_ratio
        })
        
        # Create metrics
        logger.info(f"PureMLP Training R² Score: {train_r2:.4f}")
        logger.info(f"PureMLP Test R² Score: {r2:.4f}")
        logger.info(f"PureMLP Overfit Ratio: {overfit_ratio:.4f}")
        
        metrics = {
            "Model": "PureMLP",
            "R² Score": r2,
            "MAE": mae,
            "MSE": mse,
            "Time (s)": time.time() - start_time,
            "Epochs": MODEL_PARAMS['epochs'],
            "Final LR": MODEL_PARAMS['learning_rate'],
            "Train R²": train_r2,
            "Overfit Ratio": overfit_ratio
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during PureMLP evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
