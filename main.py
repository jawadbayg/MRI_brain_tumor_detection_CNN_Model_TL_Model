from src.data_loader import load_data
from src.model_builder import build_cnn_model
from src.trainer import train_model
from src.evaluator import evaluate_model, plot_metrics
import os

if __name__ == "__main__":
    train_path = "dataset/Training"
    test_path = "dataset/Testing"

    train_gen, val_gen, test_gen = load_data(train_path, test_path)

    model = build_cnn_model()
    history = train_model(model, train_gen, val_gen, epochs=15)

    evaluate_model(model, test_gen)
    plot_metrics(history)

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    model.save('models/brain_tumor_model.h5')
