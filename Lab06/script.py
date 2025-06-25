import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot_advanced_comparison(csv_files, labels, output_dir="advanced_comparison_plots"):
    """
    Genera gráficos comparativos avanzados entre múltiples archivos CSV de entrenamiento

    Args:
        csv_files (list): Lista de rutas a archivos CSV
        labels (list): Nombres para cada modelo (mismo orden que csv_files)
        output_dir (str): Directorio donde guardar los gráficos
    """
    # Configuración de estilo
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = [14, 8]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11

    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Paleta de colores consistente
    palette = sns.color_palette("colorblind", len(csv_files))

    # Gráfico comparativo de Loss con zoom en primeras épocas
    plt.figure(figsize=(14, 6))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        plt.plot(df["epoch"], df["loss"], label=label, linewidth=2, color=palette[i])

    plt.xlabel("Época")
    plt.ylabel("Pérdida (Loss)")
    plt.title("Comparación de Pérdida durante el Entrenamiento")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_loss.png", dpi=300, bbox_inches="tight")

    # Zoom en primeras 20 épocas
    plt.xlim(0, 20)
    plt.title("Comparación de Pérdida (Primeras 20 Épocas)")
    plt.savefig(f"{output_dir}/comparison_loss_zoom.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico comparativo de Accuracy de Entrenamiento
    plt.figure(figsize=(14, 6))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        plt.plot(
            df["epoch"], df["accuracy"], label=label, linewidth=2, color=palette[i]
        )

    plt.xlabel("Época")
    plt.ylabel("Precisión (Accuracy)")
    plt.title("Comparación de Precisión en Entrenamiento")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/comparison_train_accuracy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Gráfico comparativo de Accuracy de Test
    plt.figure(figsize=(14, 6))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        plt.plot(
            df["epoch"], df["test_accuracy"], label=label, linewidth=2, color=palette[i]
        )

    plt.xlabel("Época")
    plt.ylabel("Precisión (Accuracy)")
    plt.title("Comparación de Precisión en Conjunto de Prueba")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/comparison_test_accuracy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Gráfico combinado de Accuracy (Train vs Test) para cada modelo
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for i, (df, label) in enumerate(zip(dfs, labels)):
        axes[i].plot(
            df["epoch"], df["accuracy"], label="Train", linewidth=2, color=palette[0]
        )
        axes[i].plot(
            df["epoch"],
            df["test_accuracy"],
            label="Test",
            linewidth=2,
            color=palette[1],
        )
        axes[i].set_title(f"{label} - Precisión Train vs Test")
        axes[i].set_xlabel("Época")
        axes[i].set_ylabel("Precisión")
        axes[i].yaxis.set_major_formatter(PercentFormatter(1.0))
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/train_vs_test_accuracy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Gráfico de Brecha de Generalización (Train Accuracy - Test Accuracy)
    plt.figure(figsize=(14, 6))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        generalization_gap = df["accuracy"] - df["test_accuracy"]
        plt.plot(
            df["epoch"], generalization_gap, label=label, linewidth=2, color=palette[i]
        )

    plt.xlabel("Época")
    plt.ylabel("Brecha de Generalización")
    plt.title("Brecha de Generalización (Train Accuracy - Test Accuracy)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/generalization_gap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico de Evolución Temporal
    plt.figure(figsize=(14, 6))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        plt.plot(
            df["epoch"],
            df["time_elapsed"] / 1000,
            label=label,
            linewidth=2,
            color=palette[i],
        )

    plt.xlabel("Época")
    plt.ylabel("Tiempo por Época (segundos)")
    plt.title("Comparación de Tiempo de Entrenamiento por Época")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Tabla resumen de métricas finales
    summary_data = []
    for df, label in zip(dfs, labels):
        final_train_acc = df["accuracy"].iloc[-1]
        final_test_acc = df["test_accuracy"].iloc[-1]
        final_loss = df["loss"].iloc[-1]
        avg_time = df["time_elapsed"].mean() / 1000  # en segundos

        summary_data.append(
            {
                "Modelo": label,
                "Train Accuracy": f"{final_train_acc:.2%}",
                "Test Accuracy": f"{final_test_acc:.2%}",
                "Final Loss": f"{final_loss:.4f}",
                "Tiempo/Época (s)": f"{avg_time:.2f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/summary_metrics.csv", index=False)

    print(f"Gráficos comparativos guardados en: {output_dir}")


if __name__ == "__main__":
    # Configuración
    csv_files = ["trainingLogAdamNoL2.csv", "trainingLogAdamL2Drop.csv"]
    labels = ["Modelo Base (sin regularización)", "Modelo con L2 + Dropout"]
    output_dir = "advanced_comparison_plots"

    # Generar gráficos
    plot_advanced_comparison(csv_files, labels, output_dir)
