# Clasificacion de Imagenes: IA vs Real

Proyecto de vision por computador orientado a la clasificacion binaria de imagenes generadas por inteligencia artificial frente a imagenes reales. El trabajo utiliza tecnicas de deep learning, preprocesamiento robusto de imagenes y modelos convolucionales basados en PyTorch para construir una linea base reproducible y escalable.

---

## Tabla de contenido

1. [Resumen del proyecto](#resumen-del-proyecto)
2. [Objetivo](#objetivo)
3. [Alcance y motivacion](#alcance-y-motivacion)
4. [Dataset](#dataset)
5. [Preprocesamiento](#preprocesamiento)
6. [Modelos y arquitectura seleccionada](#modelos-y-arquitectura-seleccionada)
7. [Entrenamiento](#entrenamiento)
8. [Metricas de evaluacion](#metricas-de-evaluacion)
9. [Uso de inteligencia artificial](#uso-de-inteligencia-artificial)
10. [Estructura del repositorio](#estructura-del-repositorio)
11. [Guia de ejecucion](#guia-de-ejecucion)
12. [Consideraciones tecnicas](#consideraciones-tecnicas)
13. [Referencias](#referencias)

---

## Resumen del proyecto

Los modelos generativos actuales pueden producir imagenes altamente realistas, lo que introduce retos importantes en autenticidad digital, moderacion de contenido, verificacion de informacion y analisis forense. En este proyecto se construye un clasificador binario capaz de distinguir si una imagen fue generada por IA o si corresponde a una imagen real.

El desarrollo incluye:

- carga del dataset desde Hugging Face;
- filtrado de imagenes corruptas;
- transformaciones y normalizacion para vision por computador;
- entrenamiento de una CNN base y preparacion de una arquitectura mas robusta mediante transfer learning;
- evaluacion con metricas de clasificacion estandar.

---

## Objetivo

Construir un modelo de clasificacion binaria que reciba una imagen $x$ y prediga una etiqueta $y$ tal que:

- `0`: imagen generada por IA;
- `1`: imagen real.

Formalmente, el problema se expresa como una funcion de clasificacion:

$$
f(x) \rightarrow y
$$

donde el modelo estima la probabilidad de pertenencia a una de las dos clases.

---

## Alcance y motivacion

Este proyecto tiene aplicacion en escenarios donde la deteccion de contenido sintetico es relevante:

- prevencion de desinformacion y deepfakes;
- apoyo a procesos de verificacion de contenido digital;
- analisis forense de imagenes;
- herramientas de moderacion para redes sociales, medios y plataformas creativas.

El notebook principal del repositorio implementa una solucion reproducible que sirve tanto como baseline academico como punto de partida para experimentos futuros con arquitecturas mas avanzadas.

---

## Dataset

Se utiliza el dataset [AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets).

### Informacion general

- **Fuente:** Hugging Face Datasets
- **Nombre:** `Hemg/AI-Generated-vs-Real-Images-Datasets`
- **Tamano aproximado:** 152000 imagenes
- **Peso aproximado:** 1.8 GB
- **Formato de almacenamiento:** Parquet
- **Clases:**
  - `0`: IA
  - `1`: Real

### Campos principales

- `image`: contenido de la imagen
- `label`: etiqueta binaria asociada

### Contenido visual

El dataset contiene variedad de escenas y estilos visuales, incluyendo:

- retratos;
- paisajes;
- animales;
- arte digital;
- fotografias y composiciones sinteticas.

Esta diversidad hace que el problema no sea trivial y justifica el uso de arquitecturas convolucionales con capacidad de generalizacion.

---

## Preprocesamiento

El pipeline de preprocesamiento implementado en el notebook esta pensado para ser robusto frente a inconsistencias del dataset y compatible con entrenamiento en PyTorch.

### 1. Carga del dataset

El conjunto de datos se descarga directamente desde Hugging Face mediante `load_dataset`.

```python
from datasets import load_dataset

dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
```

La particion utilizada en el notebook corresponde a `dataset['train']`.

### 2. Filtrado de imagenes corruptas

Durante el desarrollo se identificaron imagenes defectuosas que producian errores de lectura con PIL. Para resolverlo se incorporo un filtrado explicito que:

- valida cada imagen antes de usarla;
- descarta ejemplos corruptos o incompletos;
- evita la decodificacion automatica de Hugging Face cuando conviene controlar la lectura manualmente;
- carga los datos mediante un `CustomImageDataset` capaz de leer desde `bytes` o `path`.

Este paso fue fundamental para estabilizar el flujo de entrenamiento.

### 3. Redimensionamiento

Todas las imagenes se ajustan a una resolucion fija de `224 x 224` pixeles para garantizar compatibilidad con la CNN base y con modelos preentrenados como ResNet18 y EfficientNet.

### 4. Normalizacion

Se utiliza la normalizacion estandar de ImageNet:

- media: `[0.485, 0.456, 0.406]`
- desviacion estandar: `[0.229, 0.224, 0.225]`

Esta decision es especialmente importante para transfer learning sobre modelos preentrenados en ImageNet.

### 5. Data augmentation

Sobre el conjunto de entrenamiento se aplican transformaciones orientadas a mejorar la capacidad de generalizacion:

- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `ColorJitter(brightness=0.2, contrast=0.2)`

### 6. Transformaciones para validacion y prueba

Para validacion y test se usa un pipeline determinista:

- redimensionamiento a `224 x 224`;
- conversion a tensor;
- normalizacion con parametros de ImageNet.

### 7. Division del dataset

El conjunto limpio se divide en tres particiones:

- entrenamiento: 80 %
- validacion: 10 %
- prueba: 10 %

La implementacion usa `train_test_split` con semilla fija para favorecer la reproducibilidad de los experimentos.

### 8. Compatibilidad con Windows y notebooks

Debido a restricciones comunes de multiprocessing en Jupyter sobre Windows, los `DataLoader` se configuran con:

- `num_workers=0`

Esto evita fallos de workers al entrenar desde el notebook.

---

## Modelos y arquitectura seleccionada

El proyecto contempla tanto una arquitectura base como una arquitectura mas potente para experimentos principales.

### 1. Modelo base: SimpleCNN

Se implementa una CNN sencilla como baseline para verificar el pipeline completo de datos, entrenamiento y evaluacion.

#### Estructura de la red

- tres bloques convolucionales con `Conv2d + ReLU + MaxPool2d`;
- un clasificador fully connected con `Flatten`, capa densa intermedia, `Dropout(0.5)` y salida sigmoide.

#### Configuracion implementada

- `Conv2d(3, 32, kernel_size=3, padding=1)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- `Conv2d(64, 128, kernel_size=3, padding=1)`
- `Linear(128 * 28 * 28, 128)`
- `Linear(128, 1)`
- activacion final `Sigmoid()`

#### Rol en el proyecto

- sirve como linea base de referencia;
- permite validar rapidamente la integracion del pipeline;
- facilita comparar mejoras frente a modelos mas complejos.

### 2. Modelos propuestos para mejora

Con base en la naturaleza del problema, se consideran arquitecturas mas robustas:

- `ResNet18` o `ResNet34`;
- `EfficientNet-B0`.

Estas arquitecturas ofrecen mayor capacidad para aprender patrones visuales complejos y tienden a generalizar mejor sobre datasets heterogeneos.

### 3. Transfer learning

El proyecto tambien contempla transfer learning a partir de modelos preentrenados en ImageNet. La estrategia general es:

- conservar el extractor de caracteristicas preentrenado;
- reemplazar la capa final para clasificacion binaria;
- congelar parte de las capas iniciales y ajustar las finales cuando sea necesario.

### 4. Arquitectura seleccionada

La arquitectura seleccionada para el experimento principal es **ResNet18** con pesos preentrenados en ImageNet.

#### Justificacion de la eleccion

- aprovecha conocimiento visual aprendido previamente;
- mejora la capacidad de detectar patrones finos entre imagen sintetica e imagen real;
- ofrece una buena relacion entre rendimiento y costo computacional;
- resulta mas solida que una CNN simple sin llegar al costo de modelos mas pesados.

#### Adaptacion realizada

En la implementacion, la capa final fully connected de ResNet18 se reemplaza por una salida binaria para resolver el problema de clasificacion IA vs Real.

---

## Entrenamiento

El entrenamiento se realiza en PyTorch y sigue un flujo clasico supervisado.

### Componentes principales

- modelo definido en el notebook;
- funcion de perdida para clasificacion binaria;
- optimizador Adam;
- seguimiento de perdida de entrenamiento, perdida de validacion y accuracy de validacion.

### Flujo general

1. pasar batches del `train_loader` por el modelo;
2. calcular la perdida;
3. hacer `backward()`;
4. actualizar pesos con el optimizador;
5. evaluar en validacion sin gradientes;
6. almacenar metricas por epoca.

Esta separacion permite monitorear convergencia y detectar sobreajuste durante el entrenamiento.

---

## Metricas de evaluacion

El proyecto contempla varias metricas de clasificacion para evaluar el comportamiento del modelo desde distintas perspectivas.

### Metricas incluidas

- **Accuracy:** proporcion total de predicciones correctas.
- **Loss de entrenamiento y validacion:** permite observar aprendizaje, convergencia y overfitting.
- **Classification Report:** resume `precision`, `recall` y `f1-score` por clase.
- **Matriz de confusion:** muestra los aciertos y errores entre las clases `IA` y `Real`.
- **ROC Curve y AUC:** cuantifican la capacidad del modelo para separar ambas clases en distintos umbrales.

### Herramientas utilizadas

Las metricas se calculan con funciones de `scikit-learn`, entre ellas:

- `accuracy_score`
- `classification_report`
- `confusion_matrix`
- `roc_curve`
- `auc`

Esta combinacion permite una evaluacion mas completa que el uso exclusivo de accuracy.

---

## Uso de inteligencia artificial

Este proyecto hace uso de inteligencia artificial en dos sentidos complementarios.

### 1. Como objeto de estudio

El objetivo central del trabajo es detectar imagenes generadas por IA frente a imagenes reales, por lo que la IA constituye el fenomeno analizado en el problema de clasificacion.

### 2. Como apoyo al desarrollo

Durante el desarrollo se utilizaron herramientas de asistencia basadas en IA para apoyar tareas como:

- programacion;
- depuracion;
- organizacion del flujo de trabajo;
- redaccion tecnica y documentacion.

Las decisiones finales sobre implementacion, correccion del pipeline, seleccion de arquitectura y validacion del notebook fueron revisadas y ajustadas dentro del proyecto.

---

## Estructura del repositorio

```text
Redes-Neuronales/
├── README.md
├── data/
├── docs/
└── notebooks/
    └── clasificacion_IA_vs_real.ipynb
```

### Descripcion de carpetas

- `data/`: espacio reservado para datos locales o descargas manuales.
- `docs/`: carpeta destinada a material complementario, resultados o anexos.
- `notebooks/`: contiene el notebook principal del proyecto.

---

## Guia de ejecucion

### 1. Clonar el repositorio

```bash
git clone https://github.com/Tomasposada26/Redes-Neuronales.git
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si no existe un `requirements.txt` definitivo, asegurate de contar al menos con:

- `torch`
- `torchvision`
- `datasets`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pillow`

### 3. Descargar el dataset

Opcion recomendada desde el notebook:

```python
from datasets import load_dataset

dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
```

Tambien puede descargarse manualmente desde Hugging Face y ubicarse en `data/` si se desea trabajar localmente.

### 4. Ejecutar el notebook

Abrir y ejecutar en orden el notebook:

- `notebooks/clasificacion_IA_vs_real.ipynb`

### 5. Revisar resultados

Las curvas, metricas y visualizaciones producidas durante la ejecucion pueden conservarse en `docs/` o exportarse desde el notebook segun las necesidades de la entrega.

---

## Consideraciones tecnicas

- el dataset puede contener imagenes corruptas, por lo que el filtrado previo es necesario;
- en Windows y Jupyter se recomienda `num_workers=0` para evitar errores de `DataLoader`;
- el uso de `224 x 224` esta alineado con modelos preentrenados sobre ImageNet;
- el baseline con `SimpleCNN` es util para validar el pipeline, pero la arquitectura principal recomendada es `ResNet18`.

---

## Referencias

- [HuggingFace Dataset](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
