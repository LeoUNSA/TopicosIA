import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def generate_training_plots(csv_file, output_dir="plots"):
    """
    Genera gráficos de pérdida y precisión a partir del archivo CSV de logs.

    Args:
        csv_file (str): Ruta al archivo CSV con los logs de entrenamiento
        output_dir (str): Directorio donde guardar los gráficos
    """
    # Configuración de estilo
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams["font.size"] = 12

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Leer datos
    df = pd.read_csv(csv_file)

    # Gráfico de pérdida durante el entrenamiento
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="loss", linewidth=2.5, color="crimson")
    plt.title("Curva de Pérdida durante el Entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico de precisión durante el entrenamiento
    plt.figure()
    sns.lineplot(data=df, x="epoch", y="accuracy", linewidth=2.5, color="royalblue")

    # Análisis de punto óptimo
    max_acc = df["accuracy"].max()
    max_acc_epoch = df.loc[df["accuracy"].idxmax(), "epoch"]
    final_acc = df["accuracy"].iloc[-1]

    # Líneas de referencia
    plt.axhline(y=0.9, color="green", linestyle="--", alpha=0.3, label="90% precisión")
    plt.axhline(
        y=0.95, color="purple", linestyle="--", alpha=0.3, label="95% precisión"
    )

    # Marcar punto de máxima precisión
    plt.scatter(
        x=max_acc_epoch,
        y=max_acc,
        color="red",
        s=100,
        label=f"Máxima precisión: {max_acc * 100:.2f}% (época {max_acc_epoch})",
    )

    plt.title("Curva de Precisión durante el Entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico combinado
    plt.figure(figsize=(12, 6))

    # Subgráfico de pérdida
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x="epoch", y="loss", linewidth=2, color="crimson")
    plt.title("Curva de Pérdida")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")

    # Subgráfico de precisión
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x="epoch", y="accuracy", linewidth=2, color="royalblue")
    plt.title("Curva de Precisión")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.axhline(y=0.9, color="green", linestyle="--", alpha=0.3)
    plt.axhline(y=0.95, color="purple", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico de tiempo por época
    plt.figure()
    sns.lineplot(
        data=df, x="epoch", y="time_elapsed", linewidth=2.5, color="darkorange"
    )
    plt.title("Tiempo de Cómputo por Época")
    plt.xlabel("Época")
    plt.ylabel("Tiempo (ms)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_per_epoch.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráficos guardados en el directorio '{output_dir}':")
    print(f"- loss_curve.png: Curva de pérdida durante el entrenamiento")
    print(f"- accuracy_curve.png: Curva de precisión con puntos de referencia")
    print(f"- combined_curves.png: Gráfico combinado de pérdida y precisión")
    print(f"- time_per_epoch.png: Tiempo de cómputo por época")


def main():
    parser = argparse.ArgumentParser(
        description="Generar gráficos de entrenamiento a partir de archivo CSV"
    )
    parser.add_argument(
        "csv_file", type=str, help="Ruta al archivo CSV con los logs de entrenamiento"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Directorio de salida para los gráficos",
    )
    args = parser.parse_args()

    generate_training_plots(args.csv_file, args.output)


if __name__ == "__main__":
    main()
