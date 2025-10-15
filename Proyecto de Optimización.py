# PROYECTO KUKA COMPLETO - OPTIMIZACIÓN DE TRAYECTORIAS
# Reducción de vibraciones mediante Hill Climbing

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.animation as animation
from IPython.display import HTML
import warnings

warnings.filterwarnings('ignore')

print("🎯 INICIANDO PROYECTO KUKA - SUAVIZADO DE TRAYECTORIAS")
print("=" * 60)


# ============================================================================
# 1. GENERACIÓN DE TRAYECTORIA DE PRUEBA CON VIBRACIONES
# ============================================================================

def generar_trayectoria_prueba():

    print("📈 Generando trayectoria de prueba...")

    t = np.linspace(0, 8, 300)  # 8 segundos, 300 puntos

    # Trayectoria base suave (movimiento natural del robot)
    x_base = 2 * np.sin(0.8 * t) + 0.5 * t
    y_base = 1.5 * np.cos(0.6 * t) + 0.3 * t

    # Vibraciones simuladas (problema real a resolver)
    vib_x = 0.4 * np.sin(12 * t) + 0.1 * np.random.normal(0, 0.15, len(t))
    vib_y = 0.3 * np.cos(10 * t) + 0.1 * np.random.normal(0, 0.15, len(t))

    # Combinar base + vibraciones
    x = x_base + vib_x
    y = y_base + vib_y

    trajectory = np.column_stack([x, y])

    print(f"✅ Trayectoria generada: {len(trajectory)} puntos, {t[-1]:.1f} segundos")
    return trajectory, t


# Generar datos de prueba
trajectory, times = generar_trayectoria_prueba()


# ============================================================================
# 2. ANÁLISIS INICIAL - CALCULAR VIBRACIONES (JERK)
# ============================================================================

def analizar_vibraciones(trayectoria, tiempos):
    print("📊 Analizando vibraciones de la trayectoria...")

    # Crear splines cúbicos para interpolación suave
    spline_x = CubicSpline(tiempos, trayectoria[:, 0])
    spline_y = CubicSpline(tiempos, trayectoria[:, 1])

    # Evaluar en puntos más densos para análisis preciso
    t_denso = np.linspace(tiempos[0], tiempos[-1], 1000)

    # Calcular JERK (derivada tercera - indica vibraciones)
    jerk_x = spline_x.derivative(3)(t_denso)
    jerk_y = spline_y.derivative(3)(t_denso)
    jerk_total = np.sqrt(jerk_x ** 2 + jerk_y ** 2)

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
    axes[0, 1].set_ylabel('Jerk (m/s³)')
    axes[0, 1].set_title('ANÁLISIS DE VIBRACIONES - Jerk vs Tiempo')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Gráfico 3: Aceleración vs tiempo
    axes[1, 0].plot(datos_analisis['t_denso'], datos_analisis['acc_total'],
                    'g-', linewidth=2, label='Aceleración')
    axes[1, 0].axhline(y=datos_analisis['acc_max'], color='green', linestyle='--',
                       alpha=0.7, label=f'Máximo: {datos_analisis["acc_max"]:.3f}')
    axes[1, 0].set_xlabel('Tiempo (segundos)')
    axes[1, 0].set_ylabel('Aceleración (m/s²)')
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

    # Añadir valores en las barras
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
            spline_x = CubicSpline(self.tiempos_orig, trayectoria[:, 0])
            spline_y = CubicSpline(self.tiempos_orig, trayectoria[:, 1])

            # Evaluar en puntos densos para cálculo preciso
            t_denso = np.linspace(self.tiempos_orig[0], self.tiempos_orig[-1], 500)

            # Calcular jerk (derivada tercera)
            jerk_x = spline_x.derivative(3)(t_denso)
            jerk_y = spline_y.derivative(3)(t_denso)
            jerk_total = np.sqrt(jerk_x ** 2 + jerk_y ** 2)

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
            nueva_trayectoria[idx - 1] += perturbacion * 0.5
            nueva_trayectoria[idx + 1] += perturbacion * 0.3
            nueva_trayectoria[idx - 2] += perturbacion * 0.2
            nueva_trayectoria[idx + 2] += perturbacion * 0.1

        return nueva_trayectoria

    def hill_climbing_optimizado(self, iteraciones=150, paso_inicial=0.15):
        """
        Algoritmo principal de optimización con Hill Climbing mejorado
        Features:
        - Múltiples vecinos por iteración
        - Paso adaptativo
        - Reinicios suaves
        """
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


# ============================================================================
# 5. EJECUTAR OPTIMIZACIÓN
# ============================================================================

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


# ============================================================================
# 6. VISUALIZACIÓN COMPARATIVA - ANTES vs DESPUÉS
# ============================================================================

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
    {'✅ EXCELENTE - Reducción > 60%' if mejora_porcentaje > 60 else
    '🟡 MUY BUENA - Reducción 40-60%' if mejora_porcentaje > 40 else
    '🔵 BUENA - Reducción 20-40%' if mejora_porcentaje > 20 else
    '⚪ MODERADA - Reducción < 20%'}

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


# ============================================================================
# 7. ANIMACIÓN COMPARATIVA - TRAYECTORIA EN MOVIMIENTO
# ============================================================================

def crear_animacion_comparativa(trayectoria_orig, trayectoria_opt, tiempos):
    """Crea animación que muestra ambas trayectorias en movimiento"""
    print("\n🎬 Generando animación comparativa...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Configurar límites del gráfico
    margin = 0.5
    ax.set_xlim(min(trayectoria_orig[:, 0]) - margin, max(trayectoria_orig[:, 0]) + margin)
    ax.set_ylim(min(trayectoria_orig[:, 1]) - margin, max(trayectoria_orig[:, 1]) + margin)
    ax.set_xlabel('X (metros)')
    ax.set_ylabel('Y (metros)')
    ax.set_title('🎥 ANIMACIÓN: Trayectoria Original vs Optimizada', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Elementos de la animación
    line_orig, = ax.plot([], [], 'b-', linewidth=3, alpha=0.6, label='Original')
    line_opt, = ax.plot([], [], 'r-', linewidth=2, alpha=0.9, label='Optimizada')
    point_orig, = ax.plot([], [], 'bo', markersize=8, alpha=0.8, markeredgecolor='white')
    point_opt, = ax.plot([], [], 'ro', markersize=8, alpha=0.8, markeredgecolor='white')

    # Texto informativo
    text_info = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='lower right')

    def animar(frame):
        # Mostrar progreso hasta el frame actual
        line_orig.set_data(trayectoria_orig[:frame, 0], trayectoria_orig[:frame, 1])
        line_opt.set_data(trayectoria_opt[:frame, 0], trayectoria_opt[:frame, 1])

        # Puntos actuales
        if frame > 0:
            point_orig.set_data([trayectoria_orig[frame - 1, 0]], [trayectoria_orig[frame - 1, 1]])
            point_opt.set_data([trayectoria_opt[frame - 1, 0]], [trayectoria_opt[frame - 1, 1]])

        # Actualizar texto informativo
        progreso = (frame / len(trayectoria_orig)) * 100
        text_info.set_text(f'Progreso: {progreso:.1f}%\nFrame: {frame}/{len(trayectoria_orig)}')

        return line_orig, line_opt, point_orig, point_opt, text_info

    # Crear animación
    anim = animation.FuncAnimation(fig, animar, frames=len(trayectoria_orig),
                                   interval=30, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()

    print("✅ Animación creada exitosamente!")
    return anim


# Crear animación (opcional - puede tomar unos segundos)
# animacion = crear_animacion_comparativa(trajectory, trayectoria_optimizada, times)

# ============================================================================
# 8. RESUMEN FINAL Y EXPORTACIÓN
# ============================================================================

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


# Opcional: Guardar resultados
def guardar_resultados(trayectoria_opt, optimizador, filename="resultado_kuka.npz"):
    """Guarda los resultados para uso futuro"""
    np.savez(filename,
             trayectoria_optimizada=trayectoria_opt,
             historial_jerk=optimizador.historial_jerk,
             historial_mejoras=optimizador.historial_mejoras,
             mejora_porcentaje=mejora_porcentaje,
             parametros_optimizacion={
                 'iteraciones': len(optimizador.historial_jerk),
                 'jerk_inicial': jerk_orig,
                 'jerk_final': jerk_optimizado
             })
    print(f"💾 Resultados guardados en: {filename}")


# Guardar resultados (opcional)
# guardar_resultados(trayectoria_optimizada, optimizador)

print("\n✨ ¡Proyecto Terminado! ✨")