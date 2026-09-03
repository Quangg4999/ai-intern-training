"""Week 2, exercise 2.1: Linear Regression with sklearn and Gradient Descent."""

import os
from pathlib import Path

# Save figures without requiring a desktop GUI; this also works in CI and terminals.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


SEED = 42
N_SAMPLES = 200
TRUE_W = 3.5
TRUE_B = 20.0
NOISE_STD = 20.0
TEST_SIZE = 0.20
EXPERIMENT_EPOCHS = 1_000
FINAL_EPOCHS = 100_000
LEARNING_RATES = [0.000001, 0.00001, 0.0001, 0.001]
BEST_LEARNING_RATE = 0.0001


def make_data():
    """Create the deterministic dataset specified in the assignment."""
    np.random.seed(SEED)
    x = np.linspace(0, 100, N_SAMPLES)
    y = TRUE_W * x + TRUE_B + np.random.normal(0, NOISE_STD, N_SAMPLES)
    return x, y


def predict(x, w, b):
    """Predict y from the line y = w*x + b."""
    return w * x + b


def mse(y_true, y_pred):
    """Mean Squared Error implemented without sklearn."""
    return np.mean((y_true - y_pred) ** 2)


def compute_gradients(x, y_true, w, b):
    """Return gradients of MSE with respect to w and b."""
    errors = predict(x, w, b) - y_true
    n = len(x)
    dw = (2 / n) * np.sum(x * errors)
    db = (2 / n) * np.sum(errors)
    return dw, db


def train(x, y_true, learning_rate, epochs, w=0.0, b=0.0):
    """Train a line with batch Gradient Descent; stop safely on divergence."""
    loss_history = [mse(y_true, predict(x, w, b))]
    diverged = False

    for _ in range(epochs):
        dw, db = compute_gradients(x, y_true, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        current_loss = mse(y_true, predict(x, w, b))
        loss_history.append(current_loss)

        if not np.isfinite(current_loss) or current_loss > 1e100:
            diverged = True
            break

    return w, b, np.array(loss_history), diverged


def fit_sklearn(x_train, y_train, x_test, y_test):
    """Fit and evaluate sklearn's closed-form/optimized implementation."""
    model = LinearRegression()
    model.fit(x_train.reshape(-1, 1), y_train)
    train_pred = model.predict(x_train.reshape(-1, 1))
    test_pred = model.predict(x_test.reshape(-1, 1))
    return {
        "model": model,
        "w": float(model.coef_[0]),
        "b": float(model.intercept_),
        "train_mse": float(mean_squared_error(y_train, train_pred)),
        "test_mse": float(mean_squared_error(y_test, test_pred)),
    }


def run_learning_rate_experiments(x_train, y_train, x_test, y_test):
    """Train the required learning rates for the required minimum 1,000 epochs."""
    results = []
    for learning_rate in LEARNING_RATES:
        w, b, loss_history, diverged = train(
            x_train, y_train, learning_rate=learning_rate, epochs=EXPERIMENT_EPOCHS
        )
        test_mse = np.nan if diverged else mse(y_test, predict(x_test, w, b))
        results.append(
            {
                "learning_rate": learning_rate,
                "w": w,
                "b": b,
                "train_mse": loss_history[-1],
                "test_mse": test_mse,
                "epochs_completed": len(loss_history) - 1,
                "diverged": diverged,
                "loss_history": loss_history,
            }
        )
    return results


def plot_regression(x_train, y_train, x_test, y_test, sklearn_result, gd_w, gd_b, output_path):
    """Plot data and both fitted regression lines."""
    line_x = np.linspace(0, 100, 300)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label="Train", alpha=0.7, color="#1f77b4")
    plt.scatter(x_test, y_test, label="Test", alpha=0.7, color="#ff7f0e")
    plt.plot(
        line_x,
        predict(line_x, sklearn_result["w"], sklearn_result["b"]),
        label="sklearn LinearRegression",
        linewidth=2.5,
        color="#2ca02c",
    )
    plt.plot(
        line_x,
        predict(line_x, gd_w, gd_b),
        label=f"Gradient Descent (lr={BEST_LEARNING_RATE})",
        linewidth=2.5,
        linestyle="--",
        color="#d62728",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression: data and fitted lines")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_loss(experiments, output_path):
    """Show stable learning rates and the diverging rate on readable scales."""
    fig, (ax_stable, ax_diverge) = plt.subplots(1, 2, figsize=(13, 5.5))
    for result in experiments:
        safe_loss = np.clip(result["loss_history"], 1e-12, 1e100)
        label = f"lr={result['learning_rate']:g}"
        if result["diverged"]:
            ax_diverge.plot(range(len(safe_loss)), safe_loss, color="#d62728", label=label)
        else:
            ax_stable.plot(range(len(safe_loss)), safe_loss, label=label)

    ax_stable.set_yscale("log")
    ax_stable.set_xlabel("Epoch")
    ax_stable.set_ylabel("Train MSE (log scale)")
    ax_stable.set_title("Stable learning rates")
    ax_stable.grid(alpha=0.25)
    ax_stable.legend()

    ax_diverge.set_yscale("log")
    ax_diverge.set_xlabel("Epoch")
    ax_diverge.set_ylabel("Train MSE (log scale)")
    ax_diverge.set_title("Learning rate that diverges")
    ax_diverge.grid(alpha=0.25)
    ax_diverge.legend()

    fig.suptitle("Gradient Descent loss by learning rate")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_results(sklearn_result, experiments, final_gd):
    print("Dataset: y = 3.5*x + 20 + Gaussian noise (std=20)")
    print(f"Seed={SEED}; split: 80% train / 20% test; split random_state={SEED}")
    print("\n[scikit-learn]")
    print(f"w = {sklearn_result['w']:.6f}")
    print(f"b = {sklearn_result['b']:.6f}")
    print(f"Train MSE = {sklearn_result['train_mse']:.6f}")
    print(f"Test MSE  = {sklearn_result['test_mse']:.6f}")
    print("\n[Gradient Descent experiments]")
    print("lr\t\tw\t\tb\t\ttrain_mse\ttest_mse\tstatus")
    for result in experiments:
        status = "diverged" if result["diverged"] else "completed"
        print(
            f"{result['learning_rate']:.6g}\t"
            f"{result['w']:.6f}\t{result['b']:.6f}\t"
            f"{result['train_mse']:.6f}\t{result['test_mse']:.6f}\t{status}"
        )
    print("\n[Gradient Descent final run]")
    print(f"lr = {BEST_LEARNING_RATE}; epochs = {FINAL_EPOCHS}")
    print(f"w = {final_gd['w']:.6f}")
    print(f"b = {final_gd['b']:.6f}")
    print(f"Train MSE = {final_gd['train_mse']:.6f}")
    print(f"Test MSE  = {final_gd['test_mse']:.6f}")


def main():
    x, y = make_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=SEED
    )
    sklearn_result = fit_sklearn(x_train, y_train, x_test, y_test)
    experiments = run_learning_rate_experiments(x_train, y_train, x_test, y_test)
    final_w, final_b, final_loss_history, final_diverged = train(
        x_train, y_train, learning_rate=BEST_LEARNING_RATE, epochs=FINAL_EPOCHS
    )
    if final_diverged:
        raise RuntimeError("The selected learning rate diverged in the final run.")
    final_gd = {
        "w": final_w,
        "b": final_b,
        "train_mse": final_loss_history[-1],
        "test_mse": mse(y_test, predict(x_test, final_w, final_b)),
    }

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    plot_regression(
        x_train, y_train, x_test, y_test, sklearn_result, final_gd["w"], final_gd["b"],
        output_dir / "regression_line.png",
    )
    plot_loss(experiments, output_dir / "loss_by_learning_rate.png")
    print_results(sklearn_result, experiments, final_gd)


if __name__ == "__main__":
    main()
