Data: ADReSSo21 

We acknowledge the NIA AG03705 and AG05133 grants for supporting the development of DementiaBank, Pitt Corpus \cite{adressogrant}.

@article{adressogrant,
    author = {Becker, James T. and Boiler, François and Lopez, Oscar L. and Saxton, Judith and McGonigle, Karen L.},
    title = {The Natural History of Alzheimer's Disease: Description of Study Cohort and Accuracy of Diagnosis},
    journal = {Archives of Neurology},
    volume = {51},
    number = {6},
    pages = {585-594},
    year = {1994},
    month = {06},
    issn = {0003-9942},
    doi = {10.1001/archneur.1994.00540180063015},
    url = {https://doi.org/10.1001/archneur.1994.00540180063015},
    eprint = {https://jamanetwork.com/journals/jamaneurology/articlepdf/592905/archneur_51_6_015.pdf},
}

## Pipeline experimental POS

Este repositorio implementa una primera fase experimental reproducible para estudiar la **información predictiva**, la **suficiencia** de categorías gramaticales y la **degradación de rendimiento bajo ablación** en ADReSSo21. El diseño permite medir asociaciones; un mejor resultado no demuestra que BERT use causalmente una categoría.

### Datos y preparación

La distribución presente contiene `data_src/train.csv` (166 participantes; 79 control y 87 Alzheimer) y `data_src/test.csv` (71 participantes; 36 control y 35 Alzheimer), con columnas `File`, `Transcription` y `Label`. El loader detecta aliases habituales de texto, participante y etiqueta, aunque pueden fijarse explícitamente en YAML. `File` sin extensión se utiliza como `participant_id`. El test oficial se conserva separado y jamás interviene en ajuste, selección del checkpoint o decisiones experimentales.

El etiquetado usa `en_core_web_sm` de spaCy y Universal POS. Se conserva el token original, POS, posición, participante, etiqueta y split en `outputs/cache/pos_tokens.parquet`. No se eliminan stopwords, no se lematiza y se mantienen disfluencias. Los textos vacíos después de un filtro se registran mediante `was_empty` y se entregan al modelo como `[UNK]`.

Preparación solamente:

```bash
python scripts/run_experiments.py --config configs/adresso_pos.yaml --prepare-only
```

Esto crea folds `StratifiedGroupKFold` persistentes, comprueba que los participantes no se solapen, cachea POS y genera `ALL`, `POS_SEQUENCE`, todas las variantes `ONLY_*`/`WITHOUT_*`, y sus controles `RANDOM_ONLY_MATCHED_*`/`RANDOM_REMOVE_MATCHED_*` para cada seed. Los controles conservan el orden y tienen exactamente el mismo número de tokens conservados/eliminados por documento.

### Entrenamiento y evaluación

Las condiciones léxicas usan `BertForSequenceClassification` con `bert-base-uncased`, AdamW, selección por macro-F1 y early stopping exclusivamente en validation. Todos los parámetros, folds, categorías y seeds están en `configs/adresso_pos.yaml`. El weighting de clases sólo se activa si el cociente mayoritaria/minoritaria alcanza el umbral configurado, y la regla es idéntica para todas las variantes.

`POS_COUNTS` es una regresión logística sobre frecuencias normalizadas y ratios; `POS_SEQUENCE` es un BiLSTM pequeño con vocabulario específico de tags. Ambos usan exactamente las mismas asignaciones de fold que BERT. Se calculan accuracy, balanced accuracy, precision, recall, F1, macro-F1 y ROC-AUC. Las predicciones de varias transcripciones se promedian por participante.

Experimento completo (es costoso: condiciones × seeds de entrenamiento × folds × seeds random):

```bash
python scripts/run_experiments.py --config configs/adresso_pos.yaml
```

Una condición, seed y fold:

```bash
python scripts/run_experiments.py --config configs/adresso_pos.yaml \
  --variant ONLY_VERB_PRON --seed 42 --fold 0
```

Un control aleatorio concreto:

```bash
python scripts/run_experiments.py --config configs/adresso_pos.yaml \
  --variant RANDOM_ONLY_MATCHED_VERB_PRON --random-seed 7 --seed 42 --fold 0
```

Cada ejecución tiene un ID derivado de dataset, variante, seed random, seed de entrenamiento, fold y modelo. Un marcador `COMPLETED` permite reanudar sin repetir entrenamientos terminados. Cada directorio de `outputs/runs/` contiene configuración JSON, métricas JSON y predicciones CSV/parquet. `outputs/reports/` contiene tablas CSV/JSON, estadísticas descriptivas y gráficos experimentales. La comparación matched utiliza un p empírico no paramétrico con corrección de muestra finita; el módulo de métricas incluye además bootstrap pareado para predicciones sobre los mismos participantes.

### Extensión a nuevas hipótesis

Las categorías no están hardcodeadas en el generador. Añada una entrada y vuelva a preparar:

```yaml
hypotheses:
  - name: content_words
    include_pos: [NOUN, VERB, ADJ, ADV]
```

El sistema creará automáticamente `ONLY_CONTENT_WORDS`, `WITHOUT_CONTENT_WORDS` y ambos controles random matched. Para evitar reutilizar una caché antigua después de cambiar hipótesis, ejecute `--force-prepare`.

### Dependencias y pruebas

Las dependencias ya enumeradas en `requirements.txt` incluyen PyTorch, Transformers, spaCy, scikit-learn, parquet, plotting y pytest. Además se necesita el modelo de idioma:

```bash
python -m spacy download en_core_web_sm
pytest -q
```

Las pruebas cubren matching exacto, ablación/inclusión, reproducibilidad aleatoria, separación por participante y persistencia machine-readable.

> Nota práctica: si el modelo está ya en la caché local y se trabaja sin red, cambie `model.local_files_only` a `true`. Los experimentos científicos no se ejecutan durante la preparación ni durante los tests.
