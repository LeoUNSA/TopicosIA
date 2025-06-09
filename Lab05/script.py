import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.ticker import PercentFormatter

sns.set_theme()
# Configuración inicial
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_and_prepare_data(adam_path, rmsprop_path):
    """Carga y prepara los datos para comparación"""
    adam = pd.read_csv(adam_path)
    rmsprop = pd.read_csv(rmsprop_path)

    # Asegurar que ambos tienen el mismo número de épocas
    min_epochs = min(len(adam), len(rmsprop))
    adam = adam.head(min_epochs)
    rmsprop = rmsprop.head(min_epochs)

    # Agregar identificador de optimizador
    adam["Optimizador"] = "Adam"
    rmsprop["Optimizador"] = "RMSProp"

    # Combinar datos
    combined = pd.concat([adam, rmsprop])

    return combined, adam, rmsprop


def plot_loss_comparison(adam, rmsprop, save_path="plots"):
    """Genera gráfico comparativo de función de pérdida"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(adam["epoch"], adam["loss"], label="Adam", linewidth=2.5)
    plt.plot(rmsprop["epoch"], rmsprop["loss"], label="RMSProp", linewidth=2.5)

    plt.title("Comparación de Función de Pérdida: Adam vs RMSProp", pad=20)
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()

    # Mejorar formato
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Guardar
    path = os.path.join(save_path, "combined_loss.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de pérdida guardado en: {path}")


def plot_accuracy_comparison(adam, rmsprop, save_path="plots"):
    """Genera gráfico comparativo de precisión"""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(adam["epoch"], adam["accuracy"] * 100, label="Adam", linewidth=2.5)
    plt.plot(
        rmsprop["epoch"], rmsprop["accuracy"] * 100, label="RMSProp", linewidth=2.5
    )

    plt.title("Comparación de Precisión: Adam vs RMSProp", pad=20)
    plt.xlabel("Época")
    plt.ylabel("Precisión (%)")
    plt.legend()

    # Formato de porcentaje
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Mejorar formato
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Guardar
    path = os.path.join(save_path, "accuracy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de precisión guardado en: {path}")


def plot_time_comparison(adam, rmsprop, save_path="plots"):
    """Genera gráfico comparativo de tiempo de entrenamiento"""
    os.makedirs(save_path, exist_ok=True)

    # Calcular tiempo acumulado
    adam["cumulative_time"] = (
        adam["time_elapsed"].cumsum() / 1000
    )  # Convertir a segundos
    rmsprop["cumulative_time"] = rmsprop["time_elapsed"].cumsum() / 1000

    plt.figure(figsize=(12, 6))
    plt.plot(adam["epoch"], adam["cumulative_time"], label="Adam", linewidth=2.5)
    plt.plot(
        rmsprop["epoch"], rmsprop["cumulative_time"], label="RMSProp", linewidth=2.5
    )

    plt.title("Tiempo Acumulado de Entrenamiento", pad=20)
    plt.xlabel("Época")
    plt.ylabel("Tiempo (segundos)")
    plt.legend()

    # Mejorar formato
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Guardar
    path = os.path.join(save_path, "training_time_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de tiempo guardado en: {path}")


def plot_time_per_epoch(adam, rmsprop, save_path="plots"):
    """Genera gráfico de tiempo por época"""
    os.makedirs(save_path, exist_ok=True)

    # Calcular media móvil para suavizar
    window_size = 5
    adam["smooth_time"] = adam["time_elapsed"].rolling(window=window_size).mean()
    rmsprop["smooth_time"] = rmsprop["time_elapsed"].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(adam["epoch"], adam["smooth_time"], label="Adam", linewidth=2.5)
    plt.plot(rmsprop["epoch"], rmsprop["smooth_time"], label="RMSProp", linewidth=2.5)

    plt.title("Tiempo por Época (Media Móvil 5 épocas)", pad=20)
    plt.xlabel("Época")
    plt.ylabel("Tiempo (ms)")
    plt.legend()

    # Mejorar formato
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Guardar
    path = os.path.join(save_path, "time_per_epoch.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de tiempo por época guardado en: {path}")


def generate_summary_table(adam, rmsprop, save_path="plots"):
    """Genera una tabla resumen de resultados"""
    os.makedirs(save_path, exist_ok=True)

    summary_data = {
        "Metrica": [
            "Precisión Final (%)",
            "Pérdida Final",
            "Tiempo Total (s)",
            "Épocas",
        ],
        "Adam": [
            adam["accuracy"].iloc[-1] * 100,
            adam["loss"].iloc[-1],
            adam["time_elapsed"].sum() / 1000,
            len(adam),
        ],
        "RMSProp": [
            rmsprop["accuracy"].iloc[-1] * 100,
            rmsprop["loss"].iloc[-1],
            rmsprop["time_elapsed"].sum() / 1000,
            len(rmsprop),
        ],
    }

    df = pd.DataFrame(summary_data)

    # Guardar como imagen
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )

    # Formatear tabla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Resumen Comparativo", pad=20)

    path = os.path.join(save_path, "summary_table.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Tabla resumen guardada en: {path}")

    return df


def plot_combined_metrics(adam, rmsprop, save_path="plots"):
    """Gráfico combinado con subplots para comparación completa"""
    os.makedirs(save_path, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Gráfico 1: Pérdida
    ax1.plot(adam["epoch"], adam["loss"], label="Adam", linewidth=2)
    ax1.plot(rmsprop["epoch"], rmsprop["loss"], label="RMSProp", linewidth=2)
    ax1.set_title("Comparación de Función de Pérdida")
    ax1.set_ylabel("Pérdida")
    ax1.legend()

    # Gráfico 2: Precisión
    ax2.plot(adam["epoch"], adam["accuracy"] * 100, label="Adam", linewidth=2)
    ax2.plot(rmsprop["epoch"], rmsprop["accuracy"] * 100, label="RMSProp", linewidth=2)
    ax2.set_title("Comparación de Precisión")
    ax2.set_ylabel("Precisión (%)")
    ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax2.legend()

    # Gráfico 3: Tiempo acumulado
    ax3.plot(
        adam["epoch"], adam["time_elapsed"].cumsum() / 1000, label="Adam", linewidth=2
    )
    ax3.plot(
        rmsprop["epoch"],
        rmsprop["time_elapsed"].cumsum() / 1000,
        label="RMSProp",
        linewidth=2,
    )
    ax3.set_title("Comparación de Tiempo Acumulado")
    ax3.set_xlabel("Época")
    ax3.set_ylabel("Tiempo (s)")
    ax3.legend()

    plt.tight_layout()

    path = os.path.join(save_path, "combined_metrics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfico combinado guardado en: {path}")


def main(adam_path, rmsprop_path):
    """Función principal que genera todos los gráficos"""
    # Cargar y preparar datos
    combined, adam, rmsprop = load_and_prepare_data(adam_path, rmsprop_path)

    # Generar todos los gráficos
    plot_loss_comparison(adam, rmsprop)
    plot_accuracy_comparison(adam, rmsprop)
    plot_time_comparison(adam, rmsprop)
    plot_time_per_epoch(adam, rmsprop)
    plot_combined_metrics(adam, rmsprop)

    # Generar tabla resumen
    summary_df = generate_summary_table(adam, rmsprop)

    print("\nResumen de Resultados:")
    print(summary_df.to_markdown(index=False))


if __name__ == "__main__":
    # Configurar rutas a tus archivos CSV
    ADAM_CSV = "trainingLogAdam.csv"  # Reemplazar con tu ruta real
    RMSPROP_CSV = "trainingLogRMSProp.csv"  # Reemplazar con tu ruta real

    print("Generando gráficos comparativos...")
    main(ADAM_CSV, RMSPROP_CSV)
    print(
        "\nProceso completado. Todos los gráficos se han guardado en la carpeta 'plots'"
    )
