import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_and_save_model():
    print("Reading data...")
    df = pd.read_csv("data/clustered_data.csv")
    
    X = df.drop("cluster", axis=1)
    y = df["cluster"]
    
    print(f"Features ({len(X.columns)}):", list(X.columns))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    
    print("Training Logistic Regression model...")
    # Using the best parameters from the notebook
    best_lr_model = LogisticRegression(
        C=1000, 
        max_iter=113,
        multi_class='auto', 
        penalty='l2', 
        solver='lbfgs'
    )
    
    best_lr_model.fit(X_train, y_train)
    
    print(f"Training complete. Accuracy on test: {best_lr_model.score(X_test, y_test):.4f}")
    
    # Save model and feature names
    joblib.dump({
        "model": best_lr_model,
        "features": list(X.columns)
    }, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()
