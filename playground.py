from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def prepare_data(df):
    # Features to use
    features = ['ul_phr', 'ul_mcs', 'dl_tx', 'dl_bitrate', 'ul_tx', 'p_ue',
                'dl_mcs', 'dl_retx',  'ul_bitrate',  'ul_path_loss',
                'ul_retx',   'pusch_snr', 'turbo_decoder_avg'
                ]

    X = df[features]
    y = df['attack']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, features


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Initialize models
    rf = RandomForestClassifier(n_estimators=50, max_depth=50, min_samples_leaf=200, min_samples_split=500, random_state=42)
    xgb = XGBClassifier(n_estimators=50, max_depth=50, reg_alpha=0.2, reg_lambda=5e-4, learning_rate=0.5, random_state=42)

    # Train models
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    # Print accuracies
    print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
    print("XGBoost Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))

    return rf, xgb


def compute_feature_importance(model, X, y, features, model_name):
    importances = pd.DataFrame()

    # Built-in feature importance
    if model_name == "Random Forest":
        importance_scores = model.feature_importances_
    else:  # XGBoost
        importance_scores = model.feature_importances_

    importances['feature'] = features
    importances['built_in_importance'] = importance_scores

    # Permutation importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importances['permutation_importance'] = result.importances_mean
    importances['permutation_std'] = result.importances_std

    return importances.sort_values('permutation_importance', ascending=False)


def plot_feature_importance(importance_df, model_name, save_path=None):
    plt.figure(figsize=(12, 8))

    # Create bar plot for both importance metrics
    x = range(len(importance_df))
    width = 0.35

    plt.bar(x, importance_df['built_in_importance'], width,
            label='Built-in Importance', color='skyblue')
    plt.bar([i + width for i in x], importance_df['permutation_importance'], width,
            label='Permutation Importance', color='lightcoral')

    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(f'{model_name} Feature Importance Comparison')
    plt.xticks([i + width / 2 for i in x], importance_df['feature'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrices(rf_model, xgb_model, X_val, y_val, save_path=None):
    # Create confusion matrices
    rf_pred = rf_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)

    rf_cm = confusion_matrix(y_val, rf_pred)
    xgb_cm = confusion_matrix(y_val, xgb_pred)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(rf_cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Random Forest Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(xgb_cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_title('XGBoost Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/confusion_matrices.png')
        plt.close()
    else:
        plt.show()


def plot_permutation_importance_boxplots(rf_importance, xgb_importance, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # Sort features by median importance
    rf_sorted = rf_importance.sort_values('permutation_importance', ascending=True)
    xgb_sorted = xgb_importance.sort_values('permutation_importance', ascending=True)

    # Create horizontal box plots
    ax1.errorbar(rf_sorted['permutation_importance'], range(len(rf_sorted)),
                 xerr=rf_sorted['permutation_std'], fmt='o', capsize=5)
    ax2.errorbar(xgb_sorted['permutation_importance'], range(len(xgb_sorted)),
                 xerr=xgb_sorted['permutation_std'], fmt='o', capsize=5)

    # Customize plots
    ax1.set_title('Random Forest Permutation Importance')
    ax2.set_title('XGBoost Permutation Importance')

    ax1.set_yticks(range(len(rf_sorted)))
    ax2.set_yticks(range(len(xgb_sorted)))

    ax1.set_yticklabels(rf_sorted['feature'])
    ax2.set_yticklabels(xgb_sorted['feature'])

    ax1.set_xlabel('Permutation Importance')
    ax2.set_xlabel('Permutation Importance')

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}/permutation_importance_boxplots.png')
        plt.close()
    else:
        plt.show()


def save_classification_reports(rf_model, xgb_model, X_val, y_val, save_path=None):
    rf_pred = rf_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)

    rf_report = classification_report(y_val, rf_pred)
    xgb_report = classification_report(y_val, xgb_pred)

    if save_path:
        with open(f'{save_path}/classification_reports.txt', 'w') as f:
            f.write("Random Forest Classification Report:\n")
            f.write(rf_report)
            f.write("\n\nXGBoost Classification Report:\n")
            f.write(xgb_report)
    else:
        print("Random Forest Classification Report:")
        print(rf_report)
        print("\nXGBoost Classification Report:")
        print(xgb_report)


def analyze_feature_importance(train_df, val_df, save_path=None):
    # Create save directory if it doesn't exist
    if save_path:
        import os
        os.makedirs(save_path, exist_ok=True)

    # Prepare data
    X_train, y_train, features = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)

    # Train models
    rf_model, xgb_model = train_and_evaluate_models(X_train, X_val, y_train, y_val)

    # Compute feature importance for both models
    rf_importance = compute_feature_importance(rf_model, X_val, y_val, features, "Random Forest")
    xgb_importance = compute_feature_importance(xgb_model, X_val, y_val, features, "XGBoost")

    # Generate and save plots
    plot_feature_importance(rf_importance, "Random Forest", save_path)
    plot_feature_importance(xgb_importance, "XGBoost", save_path)
    plot_confusion_matrices(rf_model, xgb_model, X_val, y_val, save_path)
    plot_permutation_importance_boxplots(rf_importance, xgb_importance, save_path)
    save_classification_reports(rf_model, xgb_model, X_val, y_val, save_path)

    return rf_importance, xgb_importance, rf_model, xgb_model


if __name__ == '__main__':
    # Specify the path where you want to save the plots
    save_path = 'feature_importance_analysis2'
    test_df = pd.read_csv('/home/giorgos/projects/NWDAF-Anomaly-Detection/data/processed/test.csv')
    train_df, test_df = train_test_split(test_df, test_size=0.2, stratify=test_df['attack'])
    # Run the analysis
    rf_importance, xgb_importance, rf_model, xgb_model = analyze_feature_importance(train_df, test_df, save_path)

    print("\nAnalysis complete! Check the output directory for visualizations and reports.")
