# Clasificacion de Imagenes: IA vs Real

Proyecto de vision por computador orientado a la clasificacion binaria de imagenes generadas por inteligencia artificial frente a imagenes reales. Esta propuesta integra construccion de pipeline de datos, preprocesamiento robusto, entrenamiento de redes neuronales y evaluacion cuantitativa para producir una solucion replicable en entorno academico.

El valor principal del trabajo no se limita al resultado numerico final. Tambien busca demostrar una metodologia clara de experimentacion: partir de baselines sencillos, escalar a modelos preentrenados, comparar estrategias de entrenamiento y documentar decisiones tecnicas para que otros estudiantes puedan reproducir y extender el proyecto sin partir de cero.

---

## Tabla de contenido

1. [Resumen del proyecto](#resumen-del-proyecto)
2. [Objetivo](#objetivo)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Codigo de entrenamiento y evaluacion](#codigo-de-entrenamiento-y-evaluacion)
5. [Modelos preentrenados y checkpoints](#modelos-preentrenados-y-checkpoints)
6. [Dependencias del entorno](#dependencias-del-entorno)
7. [Alcance y motivacion](#alcance-y-motivacion)
8. [Dataset](#dataset)
9. [Preprocesamiento](#preprocesamiento)
10. [Modelos y arquitectura seleccionada](#modelos-y-arquitectura-seleccionada)
11. [Entrenamiento](#entrenamiento)
12. [Protocolo experimental](#protocolo-experimental)
13. [Metricas de evaluacion](#metricas-de-evaluacion)
14. [Reproducibilidad](#reproducibilidad)
15. [AI Usage Statement](#ai-usage-statement)
16. [Guia de ejecucion](#guia-de-ejecucion)
17. [Analisis de errores y limitaciones](#analisis-de-errores-y-limitaciones)
18. [Trabajo futuro](#trabajo-futuro)
19. [Consideraciones tecnicas](#consideraciones-tecnicas)
20. [Referencias](#referencias)

---

## Resumen del proyecto

En los ultimos anos, la calidad visual de los sistemas generativos ha evolucionado al punto de dificultar la distincion manual entre contenido real y contenido sintetico. Esta situacion genera retos en diferentes frentes: autenticidad digital, verificacion de informacion, moderacion de plataformas y analisis forense. Frente a este contexto, el presente proyecto propone y documenta una solucion de clasificacion binaria para identificar si una imagen corresponde a una muestra real o fue producida por modelos de IA.

La implementacion se construye en PyTorch y organiza el flujo completo de trabajo, desde la carga del conjunto de datos hasta la evaluacion final del modelo. A diferencia de un enfoque centrado unicamente en codigo, se da prioridad a la trazabilidad metodologica: se explica que se hizo, por que se hizo, bajo que supuestos y con que implicaciones tecnicas.

El pipeline desarrollado cubre:

- ingestion del dataset desde Hugging Face;
- validacion y limpieza de datos visuales;
- transformaciones para entrenamiento y validacion;
- entrenamiento de modelos base y modelos con transferencia de aprendizaje;
- evaluacion mediante metricas complementarias;
- guardado de checkpoints para continuidad experimental.

Con este enfoque, el repositorio funciona como evidencia de proyecto final y tambien como base para trabajos posteriores de investigacion o mejora aplicada.

---

## Objetivo

El objetivo principal es construir un clasificador binario que reciba una imagen de entrada y prediga su origen: IA o Real. Esta definicion se formaliza de la siguiente manera:

$$
f(x) \rightarrow y
$$

donde:

- $x$ representa una imagen preprocesada;
- $f$ representa la funcion aproximada por la red neuronal;
- $y$ representa la clase predicha.

En terminos de etiquetas:

- `0`: imagen generada por IA.
- `1`: imagen real.

Adicionalmente, el trabajo persigue objetivos secundarios importantes:

- evaluar la diferencia de desempeno entre arquitecturas sencillas y arquitecturas preentrenadas;
- documentar un flujo de entrenamiento reproducible en notebook;
- conservar pesos entrenados para evitar reinicios completos;
- establecer una base metodologica que permita extender el problema a escenarios de mayor complejidad.

Este planteamiento integra componente ingenieril, analitico y documental, alineado con expectativas de una entrega universitaria completa.

---

## Estructura del repositorio

```text
Redes-Neuronales/
├── README.md
├── requirements.txt
├── clasificacion_IA_vs_real_.ipynb
├── best_resnet18_phase1.pth
├── best_resnet18_finetuned.pth
└── FotosDL/
```

### Descripcion de artefactos

`README.md`:
Documento principal de referencia. Explica motivacion, metodologia, instrucciones de uso, criterios de evaluacion y decisiones tecnicas.

`requirements.txt`:
Listado de dependencias de Python requeridas para ejecutar el notebook y reproducir el entorno de trabajo.

`clasificacion_IA_vs_real_.ipynb`:
Notebook central del proyecto. Contiene carga de datos, preprocesamiento, entrenamiento, validacion, metricas y visualizaciones.

`best_resnet18_phase1.pth`:
Checkpoint correspondiente a fase de transferencia de aprendizaje con parte del backbone congelado.

`best_resnet18_finetuned.pth`:
Checkpoint posterior a fase de fine-tuning, orientada a mejorar adaptacion al dominio especifico IA vs Real.

`FotosDL/`:
Carpeta auxiliar de recursos visuales y material de apoyo utilizado en el desarrollo o en la presentacion del proyecto.

### Comentario sobre organizacion

La estructura se mantiene intencionalmente compacta para facilitar revision por docentes y jurados. Los archivos clave estan en raiz para reducir friccion al ejecutar o auditar la entrega.

---

## Codigo de entrenamiento y evaluacion

Todo el flujo de entrenamiento y evaluacion se concentra en el notebook `clasificacion_IA_vs_real_.ipynb`. Esta decision tiene dos ventajas practicas: primero, simplifica la navegacion de la entrega; segundo, permite observar en un solo lugar las celdas de implementacion y sus salidas (curvas, tablas, reportes y graficos).

El notebook esta estructurado de forma secuencial y coherente:

1. importacion de librerias y configuracion inicial;
2. descarga/carga del dataset;
3. limpieza y preparacion de imagenes;
4. definicion de transformaciones y particiones;
5. entrenamiento de modelos base;
6. experimentacion con transferencia de aprendizaje;
7. evaluacion final con metricas de clasificacion;
8. visualizaciones para interpretacion de resultados.

Se privilegia la claridad sobre la fragmentacion excesiva. Esto facilita que una persona externa pueda ejecutar el cuaderno en orden y entender la evolucion experimental sin depender de archivos adicionales.

### Alcance de esta seccion para la entrega

Para fines de documentacion, no es obligatorio reentrenar desde cero en cada revision. El notebook ya contiene resultados ejecutados y el repositorio incluye checkpoints que permiten validar el flujo sin costo computacional elevado.

---

## Modelos preentrenados y checkpoints

El repositorio incluye dos checkpoints entrenados que representan etapas distintas del proceso de optimizacion:

- `best_resnet18_phase1.pth`
- `best_resnet18_finetuned.pth`

### Por que es importante incluir checkpoints

Incluir checkpoints mejora significativamente la utilidad del proyecto porque:

- permite evaluar sin reentrenar todo el modelo;
- conserva evidencia de progreso entre fases;
- facilita comparar desempeno antes y despues del fine-tuning;
- permite retomar entrenamiento con nuevos hiperparametros;
- reduce tiempo de reproducibilidad para evaluadores.

### Lectura conceptual de cada fase

`phase1` representa una etapa inicial donde el modelo aprovecha conocimiento preentrenado y ajusta principalmente capas finales.

`finetuned` representa un ajuste mas profundo, en el que se desbloquean capas adicionales para adaptar mejor las representaciones al dominio especifico del problema.

Este esquema es util pedagogicamente porque permite mostrar como el rendimiento evoluciona al pasar de transferencia basica a ajuste especializado.

---

## Dependencias del entorno

El proyecto depende de librerias de aprendizaje profundo, procesamiento de datos, visualizacion y evaluacion estadistica. Todas las dependencias estan definidas en `requirements.txt` para mantener consistencia entre maquinas.

Instalacion recomendada:

```bash
pip install -r requirements.txt
```

### Dependencias principales por categoria

Entrenamiento y redes neuronales:

- `torch`
- `torchvision`
- `torchinfo`

Gestion de datos y utilidades numericas:

- `datasets`
- `numpy`
- `pandas`

Evaluacion y metricas:

- `scikit-learn`
- `scipy`

Imagen y vision por computador:

- `pillow`
- `opencv-python`

Visualizacion:

- `matplotlib`
- `seaborn`

### Recomendacion de entorno

Se sugiere usar un entorno virtual dedicado para evitar conflictos de versiones con otros proyectos locales. Esta practica mejora estabilidad y facilita soporte durante sustentacion.

---

## Alcance y motivacion

La motivacion del proyecto surge de una necesidad real: el aumento de contenido sintetico de alta calidad plantea dudas sobre autenticidad en contextos sociales, informativos y academicos. Contar con detectores automaticos no elimina el problema de raiz, pero si aporta una capa de verificacion util para procesos de analisis.

### Escenarios de aplicacion

- apoyo en verificacion de contenido multimedia;
- filtro preliminar para moderacion de plataformas;
- analisis forense digital en tareas de autenticacion;
- contexto educativo para estudiar deteccion de sinteticos.

### Alcance real del trabajo

Este proyecto se enfoca en clasificacion binaria de imagenes estaticas. No aborda identificacion de modelo generativo especifico, ni clasificacion multiclase por tipo de IA, ni analisis temporal sobre video. Aun asi, entrega una base experimental robusta que puede escalar hacia esas direcciones.

### Valor academico

Ademas del resultado tecnico, el valor academico radica en la disciplina metodologica: decisiones justificadas, pipeline reproducible y explicacion de limitaciones. Esto fortalece la calidad de la entrega mas alla de una simple demostracion de codigo funcional.

---

## Dataset

Se utiliza el dataset [AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets), disponible en Hugging Face.

### Informacion general

- Fuente: Hugging Face Datasets
- Nombre: `Hemg/AI-Generated-vs-Real-Images-Datasets`
- Tamano aproximado: 152000 imagenes
- Peso aproximado: 1.8 GB
- Formato principal: Parquet
- Etiquetas:
  - `0`: IA
  - `1`: Real

### Estructura de campos

- `image`: objeto de imagen
- `label`: clase binaria

### Importancia del dataset para el problema

La variedad de muestras es uno de los factores que vuelve util este conjunto de datos: no se limita a un unico estilo visual. Incluye retratos, paisajes, arte digital y escenas con diferentes condiciones de textura, color e iluminacion.

Esta diversidad reduce el riesgo de que el modelo aprenda reglas demasiado simples o triviales, y promueve aprendizaje de patrones de mayor generalizacion. No obstante, tambien incrementa la dificultad del problema y exige mayor rigor en limpieza y validacion.

### Riesgos de sesgo de distribucion

Como en cualquier dataset curado de forma especifica, existe posibilidad de sesgos por fuente o estilo dominante. Por ello, los resultados deben interpretarse como validos sobre la distribucion trabajada y no como garantia absoluta para cualquier escenario externo.

---

## Preprocesamiento

El preprocesamiento es una etapa critica porque determina la calidad de la entrada al modelo. En problemas de clasificacion de imagenes, pequenas inconsistencias en lectura, escala o normalizacion pueden afectar de forma significativa la estabilidad del entrenamiento.

### 1. Carga del dataset

```python
from datasets import load_dataset

dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")
```

La particion principal utilizada es `dataset['train']`, sobre la cual se aplican limpieza y divisiones posteriores.

### 2. Filtrado de imagenes corruptas

Durante exploracion se detectaron muestras problematicas que podian fallar al convertirse a tensor o al ser leidas por PIL. Se implemento validacion previa para descartar datos invalidos y evitar que errores de lectura interrumpan entrenamiento.

Este punto mejora robustez del pipeline y evita sesgo accidental producido por fallos silenciosos en procesamiento.

### 3. Redimensionamiento uniforme

Todas las imagenes se ajustan a `224 x 224` para garantizar compatibilidad con arquitecturas preentrenadas y mantener forma consistente de entrada en batches.

### 4. Conversion y normalizacion

Despues de convertir a tensor, se aplica normalizacion tipo ImageNet:

- media: `[0.485, 0.456, 0.406]`
- desviacion: `[0.229, 0.224, 0.225]`

Esta decision es coherente con uso de backbones preentrenados y suele mejorar convergencia inicial.

### 5. Data augmentation para entrenamiento

Se utilizan transformaciones estocasticas controladas:

- `RandomHorizontalFlip()`
- `RandomRotation(10)`
- `ColorJitter(brightness=0.2, contrast=0.2)`

El objetivo es aumentar variabilidad efectiva del conjunto de entrenamiento y reducir sobreajuste.

### 6. Pipeline determinista para validacion y prueba

En validacion y test se evita aleatoriedad para que las comparaciones entre epocas y modelos sean consistentes.

### 7. Division del conjunto

La particion del dataset se realiza en proporcion:

- entrenamiento: 80 %
- validacion: 10 %
- prueba: 10 %

Se emplea semilla fija para favorecer repetibilidad de resultados.

### 8. Compatibilidad en Windows

En entorno Jupyter sobre Windows se recomienda `num_workers=0` en DataLoader para evitar problemas de multiproceso que pueden afectar ejecucion del notebook.

### Reflexion metodologica

Preprocesar no es solo transformar imagenes; es controlar calidad de datos, estabilidad del flujo y consistencia estadistica entre etapas. Esta seccion tiene impacto directo en desempeno final.

---

## Modelos y arquitectura seleccionada

El proyecto recorre diferentes niveles de complejidad para construir una comparacion con sentido tecnico.

### 1. Baselines

#### SimpleCNN

Modelo convolucional ligero con bloques `Conv2d + ReLU + MaxPool2d`, seguido de capa densa y salida binaria. Su objetivo es validar rapidamente el pipeline end-to-end y establecer una linea base de rendimiento.

#### LeNet

Arquitectura clasica incorporada como segundo baseline para evaluar consistencia de resultados en modelos mas simples y de bajo costo computacional.

### 2. Modelos con transferencia de aprendizaje

Se experimenta con:

- `VGG16`
- `ResNet18`

En estos modelos, se reemplaza la cabeza de clasificacion para salida binaria y se exploran etapas de congelamiento y ajuste parcial o total.

### 3. Arquitectura destacada: ResNet18

`ResNet18` se adopta como arquitectura principal por su equilibrio entre profundidad, estabilidad y costo de entrenamiento. Las conexiones residuales ayudan a preservar flujo de gradiente y suelen favorecer convergencia frente a redes profundas tradicionales.

### 4. Criterios de seleccion

La eleccion final no depende de una unica metrica. Se consideran:

- desempeno en validacion;
- estabilidad entre epocas;
- capacidad de generalizacion en test;
- viabilidad computacional para el entorno disponible.

Este criterio compuesto permite justificar la arquitectura elegida con base tecnica y no solo por preferencia.

---

## Entrenamiento

El entrenamiento se implementa en PyTorch bajo esquema supervisado, con seguimiento de metricas por epoca para monitorear aprendizaje y detectar posibles problemas de sobreajuste.

### Componentes del entrenamiento

- funcion de perdida para clasificacion binaria;
- optimizador Adam;
- ciclos de entrenamiento y validacion;
- registro de perdida y accuracy;
- guardado de mejores pesos segun criterio de validacion.

### Flujo por epoca

1. forward de lotes de entrenamiento;
2. calculo de perdida;
3. retropropagacion (`backward`);
4. actualizacion de parametros;
5. evaluacion en validacion sin gradiente;
6. registro historico para analisis posterior.

### Control de calidad durante entrenamiento

Se analiza la brecha entre curvas de train y validacion para identificar:

- subajuste: aprendizaje insuficiente;
- sobreajuste: mejora en train sin mejora sostenida en validacion;
- inestabilidad: oscilaciones por tasa de aprendizaje o regularizacion.

### Importancia de checkpoints

Guardar checkpoints permite:

- recuperar mejor estado sin repetir entrenamiento completo;
- comparar fases de entrenamiento;
- facilitar auditoria del proceso experimental.

---

## Protocolo experimental

Para mantener comparabilidad, el proyecto aplica un protocolo comun entre modelos.

### Reglas del protocolo

- misma logica de particion de datos;
- mismas transformaciones por split;
- mismas metricas para comparar desempeno;
- seleccion de hiperparametros sin usar informacion del test;
- evaluacion final sobre conjunto separado.

### Secuencia experimental

1. ejecutar baselines para obtener punto de referencia;
2. aplicar transferencia de aprendizaje;
3. realizar fine-tuning controlado;
4. contrastar resultados con metricas y visualizaciones.

### Justificacion academica

Un protocolo estable evita comparaciones injustas donde cambia mas de una variable al mismo tiempo. Esto fortalece validez de conclusiones y mejora calidad argumentativa en el informe.

---

## Metricas de evaluacion

La evaluacion utiliza metricas complementarias porque la accuracy por si sola puede ocultar comportamientos relevantes, especialmente cuando hay errores asimetricos entre clases.

### Metricas reportadas

- Accuracy global.
- Perdida de entrenamiento y validacion.
- Classification report (`precision`, `recall`, `f1-score`).
- Matriz de confusion.
- Curva ROC y AUC.

### Como interpretar cada metrica

Accuracy:
Resume porcentaje total de aciertos, pero no explica distribucion de errores por clase.

Precision:
Indica pureza de predicciones positivas; util cuando interesa minimizar falsos positivos.

Recall:
Mide cobertura de positivos reales; util cuando interesa reducir falsos negativos.

F1-score:
Equilibra precision y recall en una sola medida.

Matriz de confusion:
Permite observar exactamente donde se equivoca el modelo y si existe tendencia de sesgo hacia una clase.

AUC-ROC:
Evalua capacidad de separacion de clases bajo distintos umbrales, ofreciendo una vision mas robusta del comportamiento probabilistico.

### Conclusiones a partir de metricas

La combinacion de estas metricas permite argumentar rendimiento de forma mas completa y evita conclusiones simplistas basadas en un unico numero.

---

## Reproducibilidad

La reproducibilidad es un criterio central del proyecto. Se busca que otra persona pueda repetir el flujo y obtener resultados consistentes en tendencia, aunque existan pequenas variaciones esperables por aleatoriedad del entrenamiento.

### Elementos que favorecen reproducibilidad

- notebook unico con pipeline completo;
- dependencias declaradas en `requirements.txt`;
- checkpoints entrenados disponibles;
- instrucciones de ejecucion documentadas;
- separacion explicita entre etapas de entrenamiento y evaluacion.

### Recomendaciones practicas

- fijar semillas en `random`, `numpy` y `torch`;
- documentar version de Python y sistema operativo;
- mantener registro de hiperparametros por corrida;
- evitar mezclar validacion con test;
- conservar historico de metricas por epoca.

### Sobre la variacion entre corridas

En deep learning es normal que dos corridas no sean identicas al decimal, incluso con configuraciones similares. Lo importante es la estabilidad cualitativa de resultados y la consistencia en ranking relativo de modelos.

---

## AI Usage Statement

Durante el desarrollo se utilizaron herramientas de IA como apoyo de productividad tecnica y documental.

### AI tools used

- GitHub Copilot Chat.
- Asistentes conversacionales basados en LLM para apoyo puntual de redaccion tecnica.

### Purpose of use

- acelerar tareas repetitivas de programacion;
- apoyar depuracion y exploracion de alternativas;
- mejorar redaccion y estructura de documentacion.

### Project parts supported by AI

- coding;
- debugging;
- documentation;
- organizacion del flujo experimental.

### Statement of responsibility

Todo el contenido entregado fue revisado, validado y comprendido por los autores del proyecto. Las herramientas de IA se usaron como apoyo, no como sustituto de criterio tecnico ni de responsabilidad academica.

---

## Guia de ejecucion

### 1. Clonar repositorio

```bash
git clone https://github.com/Tomasposada26/Redes-Neuronales.git
cd Redes-Neuronales
```

### 2. Crear entorno virtual (recomendado)

En Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Abrir notebook principal

Archivo principal:

- `clasificacion_IA_vs_real_.ipynb`

### 5. Cargar datos

El notebook realiza la carga del dataset desde Hugging Face mediante `load_dataset(...)`. Si los datos estan en cache local, la descarga se reutiliza automaticamente.

### 6. Ejecutar en orden

Ejecutar celdas de forma secuencial de arriba hacia abajo para preservar dependencias internas del flujo.

### 7. Modos de uso sugeridos

Modo entrega/documentacion:
Usar resultados ya guardados y checkpoints existentes para revision rapida.

Modo reproduccion completa:
Reejecutar entrenamiento y evaluacion desde inicio para validar comportamiento en el entorno local.

### 8. Exportar resultados

Graficas, tablas y metricas pueden exportarse desde Jupyter o VS Code para anexos del informe final.

---

## Analisis de errores y limitaciones

Ningun clasificador de este tipo es infalible. Es importante reportar posibles fuentes de error y limites para interpretar resultados con responsabilidad tecnica.

### Patrones comunes de error

- imagenes IA muy fotorrealistas que se acercan estadisticamente al dominio real;
- imagenes reales con edicion intensa que incorporan artefactos similares a sinteticos;
- compresion agresiva y baja resolucion que destruyen senales utiles para clasificacion.

### Limitaciones del enfoque actual

- salida binaria (no detecta tipo de generador);
- no incluye analisis multimodal ni temporal;
- sensible a cambios de dominio fuera del dataset entrenado;
- dependiente de calidad de etiquetado original.

### Implicacion para uso real

Antes de desplegar en produccion, se recomienda validacion externa con datasets adicionales, pruebas de robustez y evaluacion continua para detectar degradacion por drift de datos.

---

## Trabajo futuro

El proyecto ofrece una base clara para evolucionar hacia etapas de mayor madurez tecnica.

### Mejoras de modelado

- probar arquitecturas recientes como ConvNeXt o Vision Transformer;
- explorar ensambles entre modelos para mejorar robustez;
- incorporar tecnicas de calibracion de probabilidad.

### Mejoras de analisis

- aplicar herramientas de interpretabilidad (por ejemplo Grad-CAM);
- hacer evaluacion por subcategorias visuales;
- analizar sensibilidad a perturbaciones (ruido, blur, compresion).

### Mejoras de ingenieria

- modularizar notebook en scripts reutilizables;
- incorporar seguimiento de experimentos;
- automatizar pruebas de reproducibilidad minima.

Estas lineas permitirian transformar el proyecto desde una entrega academica solida hacia un sistema mas cercano a aplicacion real.

---

## Consideraciones tecnicas

- El filtrado de imagenes invalidas no es opcional, es parte del control de calidad del pipeline.
- En Jupyter sobre Windows, `num_workers=0` suele ser la configuracion mas estable.
- El tamano `224 x 224` y la normalizacion de ImageNet son consistentes con transferencia de aprendizaje en backbones comunes.
- Baselines ligeros son utiles para validar pipeline y detectar errores tempranos antes de ejecutar modelos mas costosos.
- La diferencia entre fase congelada y fine-tuning debe interpretarse junto con curvas y metricas, no solo con accuracy final.

---

## Referencias

- [HuggingFace Dataset](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
