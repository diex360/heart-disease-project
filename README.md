# Detective Cardíaco: Análisis de Riesgo de Enfermedad Cardíaca

Proyecto de Primer Corte — Aprendizaje de Máquina No Supervisado (Fundamentos Probabilísticos del Machine Learning). Universidad de la Sabana, 2026-II.

## Problema de negocio

Una red de clínicas cardiológicas quiere entender qué evidencia clínica resulta más informativa para distinguir presencia de enfermedad cardíaca, y cómo debería "actualizar" su sospecha diagnóstica a medida que llegan resultados de distintas pruebas. Este proyecto responde esa pregunta aplicando 11 conceptos de fundamentos probabilísticos (probabilidad condicional, Teorema de Bayes, MLE, distribuciones paramétricas, esperanza/varianza, independencia/correlación, prior/posterior, entropía, entropía condicional, entropía cruzada y divergencia KL) sobre un dataset clínico real.

## Dataset

**Heart Disease Dataset** (UCI Machine Learning Repository), base de Cleveland — 303 pacientes, 14 atributos clínicos (edad, sexo, tipo de dolor de pecho, colesterol, frecuencia cardíaca máxima, etc.) y variable objetivo de presencia de enfermedad.

- **Fuente:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart disease* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- **Licencia:** CC BY 4.0
- **Tamaño final:** n=297 (tras eliminar 6 pacientes con datos faltantes en `ca`/`thal`)

## Metodología

Python (pandas, scipy, scikit-learn) sobre Jupyter Notebook. Limpieza de datos, transformación de la variable objetivo (`num` → `target` binaria), y aplicación secuencial de los 11 conceptos estadísticos, cada uno con: pregunta de negocio → fundamento estadístico → código → resultado → interpretación. Partición 75/25 estratificada para el modelo predictivo (regresión logística).

## Resumen de los 11 hallazgos

| # | Concepto | Hallazgo |
|---|---|---|
| 1 | Probabilidad condicional | P(enfermedad\|dolor asintomático) = 72.5%, vs. 46.1% de prevalencia base |
| 2 | Teorema de Bayes | P(enfermedad\|angina inducida) = 76.3%; razón de verosimilitud (LR=3.76) mayor que la del dolor de pecho (LR=3.08) |
| 3 | Verosimilitud / MLE | Colesterol: diferencia de medias pequeña entre enfermos y sanos (251.9 vs. 243.5, d=0.161) |
| 4 | Distribuciones paramétricas | sex~Bernoulli(0.68); cp~Categórica; chol/thalach~Gaussiana aproximada (con limitación documentada) |
| 5 | Esperanza y varianza | Frecuencia cardíaca máxima: 139.1 (enfermos) vs. 158.6 (sanos), diferencia muy significativa (p≈6.11×10⁻¹⁴) |
| 6 | Independencia y correlación | Sexo NO independiente de enfermedad (χ²=21.85, p<0.001); sin multicolinealidad severa entre numéricas |
| 7 | Prior y posterior | Actualización bayesiana secuencial: 46.1% → 93.0% combinando 3 evidencias (n=57); el orden no altera el resultado final |
| 8-9 | Entropía y entropía condicional | H(target)=0.996 bits (casi máxima); H(target\|cp)=0.799 bits (19.8% de ganancia de información) |
| 10 | Entropía cruzada | Modelo con todas las variables: 0.322 nats en prueba (vs. 0.690 baseline); AUC=0.936 |
| 11 | Divergencia KL | D_KL(Enfermos‖Sanos)=0.172 bits para colesterol — discriminador débil, confirmado por Cohen's d=0.161 |

**Análisis complementario (no es uno de los 11 conceptos oficiales):** se calculó el tamaño de efecto (Cohen's d) para las variables numéricas, encontrando que `ca` (vasos coloreados por fluoroscopia), `oldpeak` (depresión del ST) y `thalach` (frecuencia cardíaca máxima) presentan los mayores tamaños de efecto (d entre 0.94 y 1.05) — muy por encima del colesterol (d=0.16).

## Conclusiones y recomendaciones

1. **Priorizar `ca`, `oldpeak` y `thalach`** como señales analíticas de mayor poder discriminativo.
2. **Explorar un score secuencial prior→posterior** combinando `cp` + `exang` + `thal` — el posterior aumenta de 46.1% a 93.0%.
3. **No usar el colesterol como criterio aislado** — es el discriminador más débil del estudio, confirmado por tres medidas complementarias.
4. **Usar el sexo como factor de estratificación**, no como criterio único (Cramér's V=0.27, efecto moderado).
5. **Validar externamente antes de derivar implicaciones operativas** — estos resultados provienen de un único centro histórico (Cleveland, 1988).

> ⚠️ Este es un análisis académico de datos históricos con fines educativos. No constituye diagnóstico médico ni debe usarse para decisiones clínicas individuales. Todas las asociaciones reportadas son estadísticas y observacionales, no evidencia de causalidad.

## Estructura del repositorio

```
heart-disease-project/
│
├── data/
│   └── processed.cleveland.data      # Dataset original (UCI Cleveland, 303 pacientes)
├── notebooks/
│   └── heart_disease_project_C1.ipynb # Análisis completo: los 11 conceptos, código + interpretación
├── README.md
└── requirements.txt
```

## Cómo reproducir el análisis

```bash
git clone <URL-de-este-repositorio>
cd heart-disease-project
pip install -r requirements.txt
jupyter notebook notebooks/heart_disease_project_C1.ipynb
```

El notebook carga automáticamente `data/processed.cleveland.data` (con manejo robusto de nombre de archivo) y ejecuta de principio a fin sin errores.

## Autor

Proyecto individual — Universidad de la Sabana, Facultad de Ingeniería.
