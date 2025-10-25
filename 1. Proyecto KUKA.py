# PROYECTO KUKA COMPLETO - OPTIMIZACIÓN DE TRAYECTORIAS
# Reducción de vibraciones mediante Hill Climbing

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings

warnings.filterwarnings('ignore')

print("🎯 INICIANDO PROYECTO KUKA - SUAVIZADO DE TRAYECTORIAS")
print("=" * 60)

def generar_trayectoria_prueba():

    print("📈 Generando trayectoria de prueba...")

    t = np.linspace(0, 8, 300)

    # Trayectoria base suave (movimiento natural del robot)
    x_base = 2 * np.sin(0.8 * t) + 0.5 * t #movimiento oscilatorio en x
    y_base = 1.5 * np.cos(0.6 * t) + 0.3 * t

    # Vibraciones simuladas (problema real a resolver)
    vib_x = 0.4 * np.sin(12 * t) + 0.1 * np.random.normal(0, 0.15, len(t))#vibraciones periodicas
    vib_y = 0.3 * np.cos(10 * t) + 0.1 * np.random.normal(0, 0.15, len(t))

    # Combinar base + vibraciones
    x = x_base + vib_x
    y = y_base + vib_y

    trajectory = np.column_stack([x, y]) #combina los arrays x y y en una sola matriz 2D

    print(f"✅ Trayectoria generada: {len(trajectory)} puntos, {t[-1]:.1f} segundos")
    return trajectory, t


# Generar datos de prueba
trajectory, times = generar_trayectoria_prueba()

def analizar_vibraciones(trayectoria, tiempos):
    print("📊 Analizando vibraciones de la trayectoria...")

    # Crear splines cúbicos para interpolación suave
    spline_x = CubicSpline(tiempos, trayectoria[:, 0])#convierte tus datos discretos de X en una función suave continua.
    spline_y = CubicSpline(tiempos, trayectoria[:, 1])

    # Evaluar en puntos más densos para análisis preciso
    t_denso = np.linspace(tiempos[0], tiempos[-1], 1000)#Aquí estamos generando 1000 puntos entre el tiempo inicial (tiempos[0]) y el final (tiempos[-1]).

    # Calcular JERK (derivada tercera - indica vibraciones)
    jerk_x = spline_x.derivative(3)(t_denso)
    jerk_y = spline_y.derivative(3)(t_denso)
    jerk_total = np.sqrt(jerk_x ** 2 + jerk_y ** 2)#magnitud total del jerk en cada instante de tiempo.

    # Calcular ACELERACIÓN (derivada segunda)
    acc_x = spline_x.derivative(2)(t_denso)
    acc_y = spline_y.derivative(2)(t_denso)
    acc_total = np.sqrt(acc_x ** 2 + acc_y ** 2)

    # Calcular VELOCIDAD (derivada primera)
    vel_x = spline_x.derivative(1)(t_denso)
    vel_y = spline_y.derivative(1)(t_denso)
    vel_total = np.sqrt(vel_x ** 2 + vel_y ** 2)

    # Métricas de vibración
    jerk_promedio = np.mean(jerk_total)
    jerk_max = np.max(jerk_total)
    acc_max = np.max(acc_total)
    vel_max = np.max(vel_total)

    print("📈 MÉTRICAS DE VIBRACIÓN INICIALES:")
    print(f"   • Jerk promedio: {jerk_promedio:.4f} (entre más bajo, más suave)")
    print(f"   • Jerk máximo: {jerk_max:.4f}")
    print(f"   • Aceleración máxima: {acc_max:.4f}")
    print(f"   • Velocidad máxima: {vel_max:.4f}")

    datos_graficos = {
        't_denso': t_denso,
        'jerk_total': jerk_total,
        'acc_total': acc_total,
        'vel_total': vel_total,
        'jerk_promedio': jerk_promedio,
        'jerk_max': jerk_max,
        'acc_max': acc_max
    }

    return jerk_promedio, datos_graficos

jerk_orig, datos_orig = analizar_vibraciones(trajectory, times)

def visualizacion_inicial(trayectoria, tiempos, datos_analisis):
    print("📊 Creando visualizaciones iniciales...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfico 1: Trayectoria completa
    axes[0, 0].plot(trayectoria[:, 0], trayectoria[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trayectoria')
    axes[0, 0].plot(trayectoria[0, 0], trayectoria[0, 1], 'go', markersize=8, label='Inicio')
    axes[0, 0].plot(trayectoria[-1, 0], trayectoria[-1, 1], 'ro', markersize=8, label='Fin')
    axes[0, 0].set_xlabel('X (metros)')
    axes[0, 0].set_ylabel('Y (metros)')
    axes[0, 0].set_title('TRAYECTORIA DEL BRAZO KUKA - Vista Superior')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    # Gráfico 2: Vibraciones (Jerk) vs tiempo
    axes[0, 1].plot(datos_analisis['t_denso'], datos_analisis['jerk_total'],
                    'r-', linewidth=2, label='Jerk (vibraciones)')
    axes[0, 1].axhline(y=datos_analisis['jerk_promedio'], color='red', linestyle='--',
                       alpha=0.7, label=f'Promedio: {datos_analisis["jerk_promedio"]:.3f}')
    axes[0, 1].set_xlabel('Tiempo (segundos)')
    axes[0, 1].set_ylabel('Jerk (mm/s³)')
    axes[0, 1].set_title('ANÁLISIS DE VIBRACIONES - Jerk vs Tiempo')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Aceleración vs tiempo
    axes[1, 0].plot(datos_analisis['t_denso'], datos_analisis['acc_total'],
                    'g-', linewidth=2, label='Aceleración')
    axes[1, 0].axhline(y=datos_analisis['acc_max'], color='green', linestyle='--',
                       alpha=0.7, label=f'Máximo: {datos_analisis["acc_max"]:.3f}')
    axes[1, 0].set_xlabel('Tiempo (segundos)')
    axes[1, 0].set_ylabel('Aceleración (mm/s²)')
    axes[1, 0].set_title('ACELERACIÓN vs TIEMPO')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Gráfico 4: Resumen métricas
    metricas = ['Jerk Promedio', 'Jerk Máximo', 'Acel Máxima']
    valores = [datos_analisis['jerk_promedio'], datos_analisis['jerk_max'], datos_analisis['acc_max']]
    colores = ['red', 'darkred', 'green']

    bars = axes[1, 1].bar(metricas, valores, color=colores, alpha=0.7)
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].set_title('MÉTRICAS DE VIBRACIÓN - Resumen')
    axes[1, 1].grid(True, alpha=0.3)

    # añade etiquetas con el valor exacto encima de cada barra en el gráfico.
    for bar, valor in zip(bars, valores):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return fig

fig_inicial = visualizacion_inicial(trajectory, times, datos_orig)


# 4. ALGORITMO DE OPTIMIZACIÓN - HILL CLIMBING MEJORADO
class OptimizadorTrayectoria:

    def __init__(self, trayectoria_original, tiempos_original):
        self.trayectoria_orig = trayectoria_original
        self.tiempos_orig = tiempos_original
        self.mejor_trayectoria = trayectoria_original.copy()
        self.mejor_jerk = float('inf')
        self.historial_jerk = []
        self.historial_mejoras = []

    def calcular_jerk_trayectoria(self, trayectoria):
        try:
            spline_x = CubicSpline(self.tiempos_orig, trayectoria[:, 0])#Se crea una interpolación cúbica
            spline_y = CubicSpline(self.tiempos_orig, trayectoria[:, 1])

            # Se genera una secuencia de 500 puntos de tiempo entre el inicio y el final de la trayectoria.
            t_denso = np.linspace(self.tiempos_orig[0], self.tiempos_orig[-1], 500)

            # Calcular jerk (derivada tercera)
            jerk_x = spline_x.derivative(3)(t_denso)
            jerk_y = spline_y.derivative(3)(t_denso)
            jerk_total = np.sqrt(jerk_x ** 2 + jerk_y ** 2)#Magnitud total del jerk

            return np.mean(jerk_total)  # Jerk promedio como métrica principal

        except Exception as e:
            print(f"⚠️ Error calculando jerk: {e}")
            return float('inf')  # Si hay error, devolver valor muy alto

    def generar_vecino_inteligente(self, trayectoria_actual, paso=0.1):
        nueva_trayectoria = trayectoria_actual.copy()
        n_puntos = len(nueva_trayectoria)

        # Número de puntos a perturbar (adaptativo)
        n_perturbaciones = max(3, n_puntos // 15)

        # Seleccionar puntos aleatorios para perturbar (excluyendo extremos)
        puntos_a_perturbar = np.random.choice(range(2, n_puntos - 2),
                                              size=n_perturbaciones,
                                              replace=False)

        for idx in puntos_a_perturbar:
            # Perturbación gaussiana suave
            perturbacion = np.random.normal(0, paso, 2)
            nueva_trayectoria[idx] += perturbacion

            # Suavizar puntos adyacentes para mantener continuidad
            # Esto evita cambios bruscos en la trayectoria
            nueva_trayectoria[idx - 1] += perturbacion * 0.5#Se reduce su magnitud pero mantiene su direccion
            nueva_trayectoria[idx + 1] += perturbacion * 0.3
            nueva_trayectoria[idx - 2] += perturbacion * 0.2
            nueva_trayectoria[idx + 2] += perturbacion * 0.1

        return nueva_trayectoria

    def hill_climbing_optimizado(self, iteraciones=150, paso_inicial=0.15):
        print("\n🎯 INICIANDO OPTIMIZACIÓN CON HILL CLIMBING")
        print(f"   Iteraciones: {iteraciones}, Paso inicial: {paso_inicial}")

        # Inicialización
        trayectoria_actual = self.trayectoria_orig.copy()
        jerk_actual = self.calcular_jerk_trayectoria(trayectoria_actual)
        paso = paso_inicial

        self.mejor_trayectoria = trayectoria_actual
        self.mejor_jerk = jerk_actual
        self.historial_jerk = [jerk_actual]
        self.historial_mejoras = [0]  # Porcentaje de mejora

        # Contadores para estadísticas
        iteraciones_sin_mejora = 0
        mejoras_totales = 0

        print(f"   Jerk inicial: {jerk_actual:.4f}")

        # Bucle principal de optimización
        for iteracion in range(iteraciones):
            # Generar múltiples vecinos y elegir el mejor
            mejor_vecino = None
            mejor_jerk_vecino = float('inf')

            # Explorar 5 vecinos por iteración
            for _ in range(5):
                vecino = self.generar_vecino_inteligente(trayectoria_actual, paso)
                jerk_vecino = self.calcular_jerk_trayectoria(vecino)

                if jerk_vecino < mejor_jerk_vecino:
                    mejor_vecino = vecino
                    mejor_jerk_vecino = jerk_vecino

            # Criterio de aceptación
            if mejor_jerk_vecino < jerk_actual:
                # ✅ MEJORA ENCONTRADA
                trayectoria_actual = mejor_vecino
                mejora = jerk_actual - mejor_jerk_vecino
                jerk_actual = mejor_jerk_vecino
                mejoras_totales += 1
                iteraciones_sin_mejora = 0

                # Reiniciar paso en éxito significativo
                if mejora > jerk_actual * 0.1:
                    paso = paso_inicial

                # Actualizar mejor global
                if jerk_actual < self.mejor_jerk:
                    self.mejor_trayectoria = trayectoria_actual
                    self.mejor_jerk = jerk_actual

            else:
                # ❌ SIN MEJORA
                iteraciones_sin_mejora += 1
                # Reducir paso gradualmente
                paso *= 0.85

            # Guardar historial
            self.historial_jerk.append(jerk_actual)
            mejora_porcentual = ((self.historial_jerk[0] - jerk_actual) / self.historial_jerk[0]) * 100
            self.historial_mejoras.append(mejora_porcentual)

            # Reporte de progreso
            if iteracion % 30 == 0 or iteracion == iteraciones - 1:
                print(f"   Iteración {iteracion:3d}: Jerk = {jerk_actual:.4f} "
                      f"(Mejora: {mejora_porcentual:.1f}%)")

        # Estadísticas finales
        print(f"\n📊 ESTADÍSTICAS DE OPTIMIZACIÓN:")
        print(f"   • Mejoras aceptadas: {mejoras_totales}/{iteraciones} "
              f"({mejoras_totales / iteraciones * 100:.1f}%)")
        print(f"   • Mejora final: {self.historial_mejoras[-1]:.1f}%")
        print(f"   • Jerk final: {self.mejor_jerk:.4f}")

        return self.mejor_trayectoria

print("\n" + "=" * 60)
print("🚀 EJECUTANDO OPTIMIZACIÓN...")
print("=" * 60)

# Crear y ejecutar optimizador
optimizador = OptimizadorTrayectoria(trajectory, times)
trayectoria_optimizada = optimizador.hill_climbing_optimizado(iteraciones=150, paso_inicial=0.12)

# Calcular métricas finales
jerk_optimizado = optimizador.calcular_jerk_trayectoria(trayectoria_optimizada)
mejora_porcentaje = ((jerk_orig - jerk_optimizado) / jerk_orig) * 100

print(f"\n🎉 RESULTADOS FINALES DE OPTIMIZACIÓN:")
print(f"   • Jerk original: {jerk_orig:.4f}")
print(f"   • Jerk optimizado: {jerk_optimizado:.4f}")
print(f"   • Reducción de vibraciones: {mejora_porcentaje:.1f}%")


def crear_comparacion_completa(trayectoria_orig, trayectoria_opt, optimizador, mejora_porcentaje):
    print("\n📈 Generando gráficos comparativos...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🎯 PROYECTO KUKA - COMPARATIVA: Trayectoria Original vs Optimizada',
                 fontsize=16, fontweight='bold')

    # 1. TRAYECTORIAS SUPERPUESTAS
    axes[0, 0].plot(trayectoria_orig[:, 0], trayectoria_orig[:, 1], 'b-',
                    linewidth=3, alpha=0.6, label='Original (con vibraciones)')
    axes[0, 0].plot(trayectoria_opt[:, 0], trayectoria_opt[:, 1], 'r-',
                    linewidth=2, alpha=0.9, label='Optimizada (suave)')
    axes[0, 0].plot(trayectoria_orig[0, 0], trayectoria_orig[0, 1], 'go',
                    markersize=12, label='Inicio', markeredgecolor='black')
    axes[0, 0].plot(trayectoria_orig[-1, 0], trayectoria_orig[-1, 1], 'mo',
                    markersize=12, label='Fin', markeredgecolor='black')
    axes[0, 0].set_xlabel('X (metros)')
    axes[0, 0].set_ylabel('Y (metros)')
    axes[0, 0].set_title('COMPARACIÓN DIRECTA - TRAYECTORIAS')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    # 2. CONVERGENCIA DEL ALGORITMO
    axes[0, 1].plot(optimizador.historial_jerk, 'g-', linewidth=2, label='Jerk durante optimización')
    axes[0, 1].axhline(y=jerk_orig, color='blue', linestyle='--', linewidth=2,
                       label=f'Original: {jerk_orig:.4f}')
    axes[0, 1].axhline(y=jerk_optimizado, color='red', linestyle='--', linewidth=2,
                       label=f'Optimizado: {jerk_optimizado:.4f}')
    axes[0, 1].set_xlabel('Iteración')
    axes[0, 1].set_ylabel('Jerk Promedio')
    axes[0, 1].set_title('CONVERGENCIA - Evolución del Jerk')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. MEJORA PORCENTUAL EN TIEMPO REAL
    axes[0, 2].plot(optimizador.historial_mejoras, 'orange', linewidth=2)
    axes[0, 2].axhline(y=mejora_porcentaje, color='red', linestyle='--', linewidth=2,
                       label=f'Mejora final: {mejora_porcentaje:.1f}%')
    axes[0, 2].set_xlabel('Iteración')
    axes[0, 2].set_ylabel('Mejora (%)')
    axes[0, 2].set_title('MEJORA PORCENTUAL - Progreso')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(bottom=0)

    # 4. COMPARACIÓN DE JERK INSTANTÁNEO
    def calcular_jerk_detallado(trayectoria):
        spline_x = CubicSpline(times, trayectoria[:, 0])
        spline_y = CubicSpline(times, trayectoria[:, 1])
        t_denso = np.linspace(times[0], times[-1], 1000)
        jerk_x = spline_x.derivative(3)(t_denso)
        jerk_y = spline_y.derivative(3)(t_denso)
        return t_denso, np.sqrt(jerk_x ** 2 + jerk_y ** 2)

    t_denso, jerk_orig_detalle = calcular_jerk_detallado(trayectoria_orig)
    _, jerk_opt_detalle = calcular_jerk_detallado(trayectoria_opt)

    axes[1, 0].plot(t_denso, jerk_orig_detalle, 'b-', alpha=0.7, label='Original', linewidth=1.5)
    axes[1, 0].plot(t_denso, jerk_opt_detalle, 'r-', alpha=0.9, label='Optimizada', linewidth=1.5)
    axes[1, 0].set_xlabel('Tiempo (s)')
    axes[1, 0].set_ylabel('Jerk Instantáneo')
    axes[1, 0].set_title('JERK vs TIEMPO - Comparación Detallada')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. HISTOGRAMA DE MEJORAS INSTANTÁNEAS
    mejora_instantanea = ((jerk_orig_detalle - jerk_opt_detalle) / jerk_orig_detalle) * 100
    # Filtrar valores infinitos
    mejora_instantanea = mejora_instantanea[np.isfinite(mejora_instantanea)]

    axes[1, 1].hist(mejora_instantanea, bins=30, alpha=0.7, color='green',
                    edgecolor='black', density=True)
    axes[1, 1].axvline(x=mejora_porcentaje, color='red', linestyle='--', linewidth=2,
                       label=f'Mejora promedio: {mejora_porcentaje:.1f}%')
    axes[1, 1].set_xlabel('Reducción de Jerk (%)')
    axes[1, 1].set_ylabel('Densidad de Probabilidad')
    axes[1, 1].set_title('DISTRIBUCIÓN DE MEJORAS INSTANTÁNEAS')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. RESUMEN NUMÉRICO
    axes[1, 2].axis('off')

    # Calcular métricas adicionales
    def calcular_metricas_adicionales(trayectoria):
        spline_x = CubicSpline(times, trayectoria[:, 0])
        spline_y = CubicSpline(times, trayectoria[:, 1])
        t_denso = np.linspace(times[0], times[-1], 1000)

        vel_x = spline_x.derivative(1)(t_denso)
        vel_y = spline_y.derivative(1)(t_denso)
        vel_total = np.sqrt(vel_x ** 2 + vel_y ** 2)

        acc_x = spline_x.derivative(2)(t_denso)
        acc_y = spline_y.derivative(2)(t_denso)
        acc_total = np.sqrt(acc_x ** 2 + acc_y ** 2)

        return {
            'vel_max': np.max(vel_total),
            'acc_max': np.max(acc_total),
            'vel_promedio': np.mean(vel_total),
            'acc_promedio': np.mean(acc_total)
        }

    metricas_orig = calcular_metricas_adicionales(trayectoria_orig)
    metricas_opt = calcular_metricas_adicionales(trayectoria_opt)

    texto_resumen = f"""
    📊 RESUMEN DE RESULTADOS - PROYECTO KUKA

    🎯 EFECTO EN VIBRACIONES:
    • Reducción de Jerk: {mejora_porcentaje:.1f}%
    • Jerk original: {jerk_orig:.4f}
    • Jerk optimizado: {jerk_optimizado:.4f}

    ⚡ MÉTRICAS DE MOVIMIENTO:
    • Velocidad máxima: {metricas_opt['vel_max']:.2f} m/s
    • Aceleración máxima: {metricas_opt['acc_max']:.2f} m/s²
    • Velocidad promedio: {metricas_opt['vel_promedio']:.2f} m/s

    📈 ESTADÍSTICAS DE OPTIMIZACIÓN:
    • Iteraciones totales: {len(optimizador.historial_jerk)}
    • Puntos en trayectoria: {len(trayectoria_orig)}
    • Duración: {times[-1]:.1f} segundos

    🏆 EVALUACIÓN FINAL:
    {'EXCELENTE - Reducción > 60%' if mejora_porcentaje > 60 else
    'MUY BUENA - Reducción 40-60%' if mejora_porcentaje > 40 else
    'BUENA - Reducción 20-40%' if mejora_porcentaje > 20 else
    'MODERADA - Reducción < 20%'}

    💡 INTERPRETACIÓN:
    El algoritmo redujo significativamente las vibraciones
    manteniendo la trayectoria general del brazo robótico.
    """

    axes[1, 2].text(0.05, 0.95, texto_resumen, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.show()

    return fig


# Crear comparativa completa
fig_comparativa = crear_comparacion_completa(trajectory, trayectoria_optimizada,
                                             optimizador, mejora_porcentaje)

print("\n" + "=" * 70)
print("🎉 PROYECTO KUKA COMPLETADO EXITOSAMENTE!")
print("=" * 70)

print(f"""
📋 RESUMEN EJECUTIVO:

🎯 OBJETIVO: Reducir vibraciones en trayectoria de brazo KUKA
✅ RESULTADO: {mejora_porcentaje:.1f}% de reducción en vibraciones

📊 MÉTRICAS PRINCIPALES:
   • Vibraciones (Jerk): {jerk_orig:.4f} → {jerk_optimizado:.4f}
   • Mejora: {mejora_porcentaje:.1f}%
   • Iteraciones: {len(optimizador.historial_jerk)}
   • Duración optimizada: {times[-1]:.1f} segundos
""")


def evaluar_rutas_pruebas():
    """
    Evalúa el algoritmo con 5 trayectorias de prueba diferentes
    y analiza el jerk antes y después de la optimización.
    """
    print("\n" + "=" * 70)
    print("🧪 EVALUANDO 5 TRAYECTORIAS DE PRUEBA - ANÁLISIS DE JERK")
    print("=" * 70)

    # Configuración común
    tiempos = np.linspace(0, 8, 300)

    # Generar 5 trayectorias de prueba diferentes
    trayectorias_prueba = {
        "PRUEBA_1": generar_trayectoria_1(tiempos),
        "PRUEBA_2": generar_trayectoria_2(tiempos),
        "PRUEBA_3": generar_trayectoria_3(tiempos),
        "PRUEBA_4": generar_trayectoria_4(tiempos),
        "PRUEBA_5": generar_trayectoria_5(tiempos)
    }

    resultados = {}

    for nombre, trayectoria in trayectorias_prueba.items():
        print(f"\n🔍 Analizando: {nombre}")
        print("-" * 40)

        # Analizar vibraciones iniciales
        jerk_inicial, datos_inicial = analizar_vibraciones(trayectoria, tiempos)

        # Optimizar trayectoria
        optimizador = OptimizadorTrayectoria(trayectoria, tiempos)
        trayectoria_opt = optimizador.hill_climbing_optimizado(iteraciones=50, paso_inicial=0.1)

        # Analizar vibraciones finales
        jerk_final = optimizador.calcular_jerk_trayectoria(trayectoria_opt)
        mejora = ((jerk_inicial - jerk_final) / jerk_inicial) * 100

        # Guardar resultados
        resultados[nombre] = {
            'jerk_inicial': jerk_inicial,
            'jerk_final': jerk_final,
            'mejora_porcentaje': mejora,
            'trayectoria_original': trayectoria,
            'trayectoria_optimizada': trayectoria_opt,
            'optimizador': optimizador
        }

        print(f"   • Jerk inicial: {jerk_inicial:.4f}")
        print(f"   • Jerk final: {jerk_final:.4f}")
        print(f"   • Mejora: {mejora:.1f}%")

    # Mostrar resumen y gráficas
    mostrar_resultados_completos(resultados, tiempos)

    return resultados


def generar_trayectoria_1(tiempos):
    """Trayectoria lineal con vibraciones suaves"""
    x = 0.8 * tiempos + 0.2 * np.sin(6 * tiempos)
    y = 0.5 * tiempos + 0.1 * np.cos(8 * tiempos)
    return np.column_stack([x, y])


def generar_trayectoria_2(tiempos):
    """Trayectoria circular con vibraciones moderadas"""
    radio = 1.5
    x = radio * np.cos(0.7 * tiempos) + 0.3 * np.sin(10 * tiempos)
    y = radio * np.sin(0.7 * tiempos) + 0.2 * np.cos(12 * tiempos)
    return np.column_stack([x, y])


def generar_trayectoria_3(tiempos):
    """Trayectoria sinusoidal compleja"""
    x = 2 * np.sin(0.5 * tiempos) + 0.5 * np.sin(3 * tiempos) + 0.1 * np.random.normal(0, 0.15, len(tiempos))
    y = 1.5 * np.cos(0.6 * tiempos) + 0.4 * np.cos(2 * tiempos) + 0.1 * np.random.normal(0, 0.1, len(tiempos))
    return np.column_stack([x, y])


def generar_trayectoria_4(tiempos):
    """Trayectoria en espiral con vibraciones"""
    radio = 0.1 * tiempos
    x = radio * np.cos(2 * tiempos) + 0.2 * np.sin(15 * tiempos)
    y = radio * np.sin(2 * tiempos) + 0.15 * np.cos(18 * tiempos)
    return np.column_stack([x, y])


def generar_trayectoria_5(tiempos):
    """Trayectoria con cambios bruscos de dirección"""
    x = np.zeros_like(tiempos)
    y = np.zeros_like(tiempos)

    # Crear segmentos con diferentes direcciones
    segmentos = 5
    puntos_por_segmento = len(tiempos) // segmentos

    for i in range(segmentos):
        inicio = i * puntos_por_segmento
        fin = (i + 1) * puntos_por_segmento if i < segmentos - 1 else len(tiempos)

        if i % 2 == 0:
            x[inicio:fin] = 0.5 * tiempos[inicio:fin] + 0.3 * np.sin(8 * tiempos[inicio:fin])
            y[inicio:fin] = 0.3 * tiempos[inicio:fin] + 0.2 * np.cos(6 * tiempos[inicio:fin])
        else:
            x[inicio:fin] = 1.0 - 0.3 * tiempos[inicio:fin] + 0.2 * np.sin(10 * tiempos[inicio:fin])
            y[inicio:fin] = 0.8 - 0.2 * tiempos[inicio:fin] + 0.15 * np.cos(12 * tiempos[inicio:fin])

    return np.column_stack([x, y])


def mostrar_resultados_completos(resultados, tiempos):
    """Muestra gráficas completas de las 5 pruebas"""
    print("\n" + "=" * 70)
    print("📊 RESULTADOS COMPLETOS - 5 PRUEBAS DE TRAYECTORIAS")
    print("=" * 70)

    # Crear figura grande con subplots organizados correctamente
    fig = plt.figure(figsize=(20, 16))

    # 1. Gráfico de comparación de mejoras
    ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=2)
    nombres = list(resultados.keys())
    mejoras = [resultados[nombre]['mejora_porcentaje'] for nombre in nombres]
    jerks_inicial = [resultados[nombre]['jerk_inicial'] for nombre in nombres]
    jerks_final = [resultados[nombre]['jerk_final'] for nombre in nombres]

    # Barras de mejora
    colores = ['#2E8B57' if m > 25 else '#FFA500' if m > 15 else '#FF4500' for m in mejoras]
    bars = ax1.bar(nombres, mejoras, color=colores, alpha=0.8)
    ax1.set_ylabel('Mejora del Jerk (%)')
    ax1.set_title('COMPARACIÓN DE MEJORAS - LAS 5 PRUEBAS')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for bar, valor in zip(bars, mejoras):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{valor:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 2. Gráfico de Jerk inicial vs final
    ax2 = plt.subplot2grid((4, 5), (0, 2), colspan=2)
    x_pos = np.arange(len(nombres))
    ancho = 0.35

    ax2.bar(x_pos - ancho / 2, jerks_inicial, ancho, label='Jerk Inicial',
            alpha=0.7, color='red', edgecolor='darkred')
    ax2.bar(x_pos + ancho / 2, jerks_final, ancho, label='Jerk Final',
            alpha=0.7, color='green', edgecolor='darkgreen')

    ax2.set_ylabel('Jerk Promedio')
    ax2.set_title('JERK INICIAL vs FINAL')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(nombres, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Resumen estadístico
    ax3 = plt.subplot2grid((4, 5), (0, 4))
    ax3.axis('off')

    texto_resumen = "📈 RESUMEN ESTADÍSTICO:\n\n"
    mejora_promedio = np.mean(mejoras)
    mejora_max = np.max(mejoras)
    mejora_min = np.min(mejoras)

    texto_resumen += f"Mejora promedio: {mejora_promedio:.1f}%\n"
    texto_resumen += f"Mejora máxima: {mejora_max:.1f}%\n"
    texto_resumen += f"Mejora mínima: {mejora_min:.1f}%\n\n"
    texto_resumen += f"Pruebas exitosas: {sum(1 for m in mejoras if m > 15)}/5"

    ax3.text(0.1, 0.9, texto_resumen, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # 4. Trayectorias originales vs optimizadas (fila 1)
    for i, (nombre, datos) in enumerate(list(resultados.items())[:3]):  # Primeras 3
        ax = plt.subplot2grid((4, 5), (1, i))

        trayectoria_orig = datos['trayectoria_original']
        trayectoria_opt = datos['trayectoria_optimizada']

        ax.plot(trayectoria_orig[:, 0], trayectoria_orig[:, 1],
                'b-', linewidth=2, alpha=0.6, label='Original')
        ax.plot(trayectoria_opt[:, 0], trayectoria_opt[:, 1],
                'r-', linewidth=1.5, alpha=0.8, label='Optimizada')

        ax.set_title(f'{nombre}\nJerk: {datos["jerk_inicial"]:.1f} → {datos["jerk_final"]:.1f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # 5. Trayectorias originales vs optimizadas (fila 2 - últimas 2)
    for i, (nombre, datos) in enumerate(list(resultados.items())[3:], 3):  # Últimas 2
        ax = plt.subplot2grid((4, 5), (1, i))

        trayectoria_orig = datos['trayectoria_original']
        trayectoria_opt = datos['trayectoria_optimizada']

        ax.plot(trayectoria_orig[:, 0], trayectoria_orig[:, 1],
                'b-', linewidth=2, alpha=0.6, label='Original')
        ax.plot(trayectoria_opt[:, 0], trayectoria_opt[:, 1],
                'r-', linewidth=1.5, alpha=0.8, label='Optimizada')

        ax.set_title(f'{nombre}\nJerk: {datos["jerk_inicial"]:.1f} → {datos["jerk_final"]:.1f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # 6. Análisis de jerk detallado para cada prueba (filas 2 y 3)
    for i, (nombre, datos) in enumerate(resultados.items()):
        fila = 2 + i // 3  # Distribuir en filas 2 y 3
        columna = i % 3

        ax = plt.subplot2grid((4, 5), (fila, columna))

        # Calcular jerk instantáneo
        def calcular_jerk_instantaneo(trayectoria):
            spline_x = CubicSpline(tiempos, trayectoria[:, 0])
            spline_y = CubicSpline(tiempos, trayectoria[:, 1])
            t_denso = np.linspace(tiempos[0], tiempos[-1], 500)
            jerk_x = spline_x.derivative(3)(t_denso)
            jerk_y = spline_y.derivative(3)(t_denso)
            return t_denso, np.sqrt(jerk_x ** 2 + jerk_y ** 2)

        t_denso, jerk_orig = calcular_jerk_instantaneo(datos['trayectoria_original'])
        _, jerk_opt = calcular_jerk_instantaneo(datos['trayectoria_optimizada'])

        ax.plot(t_denso, jerk_orig, 'b-', alpha=0.6, label='Jerk Original', linewidth=1)
        ax.plot(t_denso, jerk_opt, 'r-', alpha=0.8, label='Jerk Optimizado', linewidth=1)

        ax.set_title(f'JERK INSTANTÁNEO - {nombre}')
        ax.set_xlabel('Tiempo (s)')
        if columna == 0:
            ax.set_ylabel('Jerk')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Mostrar resumen numérico
    print("\n📋 RESUMEN NUMÉRICO DE LAS 5 PRUEBAS:")
    print("-" * 50)
    for nombre, datos in resultados.items():
        print(f"• {nombre}:")
        print(f"  Jerk inicial: {datos['jerk_inicial']:8.4f}")
        print(f"  Jerk final:   {datos['jerk_final']:8.4f}")
        print(f"  Mejora:       {datos['mejora_porcentaje']:7.1f}%")
        print(
            f"  Evaluación:   {'✅ EXCELENTE' if datos['mejora_porcentaje'] > 30 else '✅ BUENA' if datos['mejora_porcentaje'] > 20 else '⚠️ MODERADA' if datos['mejora_porcentaje'] > 10 else '❌ BAJA'}")
        print()
        def calcular_jerk_instantaneo(trayectoria):
            spline_x = CubicSpline(tiempos, trayectoria[:, 0])
            spline_y = CubicSpline(tiempos, trayectoria[:, 1])
            t_denso = np.linspace(tiempos[0], tiempos[-1], 500)
            jerk_x = spline_x.derivative(3)(t_denso)
            jerk_y = spline_y.derivative(3)(t_denso)
            return t_denso, np.sqrt(jerk_x ** 2 + jerk_y ** 2)

        t_denso, jerk_orig = calcular_jerk_instantaneo(datos['trayectoria_original'])
        _, jerk_opt = calcular_jerk_instantaneo(datos['trayectoria_optimizada'])

        ax.plot(t_denso, jerk_orig, 'b-', alpha=0.6, label='Jerk Original', linewidth=1)
        ax.plot(t_denso, jerk_opt, 'r-', alpha=0.8, label='Jerk Optimizado', linewidth=1)

        ax.set_title(f'JERK INSTANTÁNEO - {nombre}')
        ax.set_xlabel('Tiempo (s)')
        if i == 0:
            ax.set_ylabel('Jerk')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Mostrar resumen numérico
    print("\n📋 RESUMEN NUMÉRICO DE LAS 5 PRUEBAS:")
    print("-" * 50)
    for nombre, datos in resultados.items():
        print(f"• {nombre}:")
        print(f"  Jerk inicial: {datos['jerk_inicial']:8.4f}")
        print(f"  Jerk final:   {datos['jerk_final']:8.4f}")
        print(f"  Mejora:       {datos['mejora_porcentaje']:7.1f}%")
        print()


# Agregar esta línea al final del código principal para ejecutar las pruebas
print("\n" + "=" * 70)
print("🧪 EJECUTANDO PRUEBAS DE ROBUSTEZ...")
print("=" * 70)

# Ejecutar las pruebas
resultados_pruebas = evaluar_rutas_pruebas()



