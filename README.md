# Screen2Tourism

Screen2Tourism is a research-oriented framework for **film location recommendation** and **film tourism route generation**. The repository contains two main computational components that use the same underlying film-tourism data but process it differently for recommendation and route-generation tasks.

The repository is organized so that the main methodological tasks can be examined and executed separately.

---

## Repository Structure

```text
Screen2Tourism/
│
├── datasets/
│   ├── location-based/
│   └── route-based/
│
├── film-tourism-route-generation/
│   ├── 01_dataset_construction.py
│   ├── 02_shortest_path_method.py
│   ├── 03_content_based_method.py
│   ├── 04_knowledge_graph_method.py
│   ├── 05_llm_based_method.py
│   ├── 06_rag_assistant.py
│   └── rag_groundness_eval.csv
│
├── location-recommendation-system/
│   ├── models/
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── categories.py
│   │   ├── entity_extraction.py
│   │   ├── genre_classifier.py
│   │   ├── graph_recommender.py
│   │   ├── ner_utils.py
│   │   ├── summarizer.py
│   │   ├── text_search.py
│   │   ├── text_utils.py
│   │   └── visual_search.py
│   ├── scripts/
│   └── .gitignore
│
└── requirements.txt
```

---

# 1. Datasets

Both main components are based on the same underlying film-tourism data. However, the data are processed differently according to the requirements of each task. For this reason, task-specific datasets are stored separately while remaining under a common `datasets/` directory.

```text
datasets/
├── location-based/
└── route-based/
```

### `datasets/location-based/`

Contains the data used by the **Location Recommendation System**. Depending on the recommendation method, these data may include textual location descriptions, production and genre information, location metadata, graph relations, and visual features or image-related data.

### `datasets/route-based/`

Contains the data used by the **Film Tourism Route Generation** methods. These files are prepared for spatial, content-based, knowledge-graph-based, and LLM-supported route-generation experiments.

This organization keeps method-specific preprocessing outputs clearly separated while preserving a common dataset layer.

---

# 2. Location Recommendation System

The Location Recommendation System identifies candidate filming locations from different types of input. Four complementary recommendation approaches are used, followed by a metadata-based filtering layer.

```text
Location Recommendation System
│
├── NER-based Location Recommendation
├── Genre-based Location Recommendation
├── Image-based Location Recommendation
├── Graph-based Location Recommendation
└── Content-based Filtering Using Metadata
```

The implementation is located under:

```text
location-recommendation-system/
```

## 2.1 NER-based Location Recommendation

The NER-based method processes screenplay text and extracts location-related information. The extracted entities and contextual information are matched against stored location profiles to identify suitable filming locations.

Relevant modules include:

```text
recommender/entity_extraction.py
recommender/ner_utils.py
recommender/text_search.py
recommender/text_utils.py
```

A screenplay summarization step can also be used before text-based recommendation:

```text
recommender/summarizer.py
```

The summarizer uses an LLM to produce a location-oriented representation of the screenplay, with emphasis on outdoor settings, cinematic atmosphere, and production-relevant location characteristics.

---

## 2.2 Genre-based Location Recommendation

The genre-based method predicts or identifies the production genre and prioritizes locations associated with productions of similar genres.

Main module:

```text
recommender/genre_classifier.py
```

The predicted genre information is combined with the available location and production metadata to support candidate ranking.

---

## 2.3 Image-based Location Recommendation

The image-based method compares a reference image with visual representations of known filming locations and retrieves visually similar candidates.

Main module:

```text
recommender/visual_search.py
```

This component supports location recommendation when the user provides visual rather than textual input.

---

## 2.4 Graph-based Location Recommendation

The graph-based method models relationships among productions and filming locations. It recommends candidate locations based on graph connectivity and previously observed co-occurrence relationships.

Main module:

```text
recommender/graph_recommender.py
```

This approach allows the system to use relational information that is not directly captured by textual or visual similarity.

---

## 2.5 Content-based Filtering Using Metadata

A metadata layer is applied after candidate generation. It can restrict recommendations according to information such as province or region and can increase the ranking of locations whose type matches the submitted text or recommendation context.

Supporting modules include:

```text
recommender/categories.py
recommender/text_utils.py
```

The overall recommendation workflow can be summarized as:

```text
                    User Input
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
 Screenplay/Text      Genre            Image
       │                │                │
       ▼                ▼                ▼
 NER-based         Genre-based      Image-based
 Recommendation    Recommendation   Recommendation
       │                │                │
       └────────────┬───┴───────────────┘
                    │
                    │        Graph Relations
                    │              │
                    │              ▼
                    │       Graph-based
                    │       Recommendation
                    │              │
                    └───────┬──────┘
                            ▼
                  Metadata-based Filtering
                            │
                            ▼
                  Recommended Locations
```

---

# 3. Film Tourism Route Generation

The Film Tourism Route Generation module consists of four complementary computational components:

```text
Film Tourism Route Generation
│
├── Shortest-path-based Route Generation
├── Content-based Route Generation
├── Knowledge Graph-based Route Generation
└── LLM-based Route Generation
```

The implementation is located under:

```text
film-tourism-route-generation/
```

These components are designed as parts of a layered architecture rather than as completely independent route-generation systems. The content-based, knowledge-graph-based, and LLM-based approaches reuse the common location-ordering and route-optimization mechanism.

---

## 3.1 Dataset Construction

```text
01_dataset_construction.py
```

This script prepares the structured dataset used by the route-generation experiments. The resulting processed data are stored under:

```text
datasets/route-based/
```

---

## 3.2 Shortest-path-based Route Generation

```text
02_shortest_path_method.py
```

This component considers spatial relationships between selected locations and determines an efficient visiting order.

**Main purpose**

- construct the spatial route representation;
- calculate distances between locations;
- determine the visiting order;
- provide the common route-optimization mechanism used by the other route-generation methods.

---

## 3.3 Content-based Route Generation

```text
03_content_based_method.py
```

The content-based method identifies candidate locations according to user preferences and then applies the common route-ordering mechanism.

**Main input:** user preferences  
**Main output:** preference-aware film-tourism route

**Dependency:**

```text
02_shortest_path_method.py
```

---

## 3.4 Knowledge Graph-based Route Generation

```text
04_knowledge_graph_method.py
```

This method uses relationships among productions, actors, directors, genres, and locations to identify candidate destinations beyond direct location attributes.

The selected destinations are subsequently ordered using the shared route-generation mechanism.

**Dependency:**

```text
02_shortest_path_method.py
```

---

## 3.5 LLM-based Route Generation

```text
05_llm_based_method.py
```

The LLM-based component transforms a natural-language user request into structured route constraints such as destination, trip duration, and user preferences.

These constraints are then passed to the existing candidate-selection and route-ordering components.

**Dependencies:**

```text
02_shortest_path_method.py
03_content_based_method.py
```

---

# 4. Retrieval-Augmented Tourism Assistant

The repository also contains a Retrieval-Augmented Tourism Assistant:

```text
06_rag_assistant.py
```

Unlike the route-generation methods, this component is designed for **question answering** rather than route creation. It retrieves relevant information from the film-tourism dataset and generates grounded responses to user queries.

The corresponding evaluation file is:

```text
film-tourism-route-generation/rag_groundness_eval.csv
```

This file contains the evaluation data used for assessing the grounding behavior of the assistant.

---

# 5. Method Overview

| Main Component | Task | Main Code |
|---|---|---|
| Location Recommendation | NER-based location recommendation | `entity_extraction.py`, `ner_utils.py`, `text_search.py` |
| Location Recommendation | Genre-based location recommendation | `genre_classifier.py` |
| Location Recommendation | Image-based location recommendation | `visual_search.py` |
| Location Recommendation | Graph-based location recommendation | `graph_recommender.py` |
| Location Recommendation | Metadata-based filtering | `categories.py`, `text_utils.py` |
| Film Tourism Route Generation | Dataset construction | `01_dataset_construction.py` |
| Film Tourism Route Generation | Shortest-path-based route generation | `02_shortest_path_method.py` |
| Film Tourism Route Generation | Content-based route generation | `03_content_based_method.py` |
| Film Tourism Route Generation | Knowledge graph-based route generation | `04_knowledge_graph_method.py` |
| Film Tourism Route Generation | LLM-based route generation | `05_llm_based_method.py` |
| Tourism Assistant | Retrieval-Augmented Tourism Assistant | `06_rag_assistant.py` |

---

# 6. Installation

Clone the repository:

```bash
git clone <repository-url>
cd Screen2Tourism
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 7. Environment Variables

Some LLM-based components use the Groq API. The API key should be provided as an environment variable and should **not** be committed to the repository.

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_api_key"
```

### Linux/macOS

```bash
export GROQ_API_KEY="your_api_key"
```

If a `.env` file is used locally, it should be excluded through `.gitignore`.

---

# 8. Running the Code

## 8.1 Location Recommendation System

The reusable recommendation modules are located under:

```text
location-recommendation-system/recommender/
```

Runnable experiment or application scripts should be executed from:

```text
location-recommendation-system/scripts/
```

Example:

```bash
python location-recommendation-system/scripts/<script_name>.py
```

The exact script depends on the recommendation task being evaluated.

---

## 8.2 Film Tourism Route Generation

The route-generation experiments can be executed separately.

### Dataset construction

```bash
python film-tourism-route-generation/01_dataset_construction.py
```

### Shortest-path-based method

```bash
python film-tourism-route-generation/02_shortest_path_method.py
```

### Content-based method

```bash
python film-tourism-route-generation/03_content_based_method.py
```

### Knowledge graph-based method

```bash
python film-tourism-route-generation/04_knowledge_graph_method.py
```

### LLM-based method

```bash
python film-tourism-route-generation/05_llm_based_method.py
```

### Retrieval-Augmented Tourism Assistant

```bash
python film-tourism-route-generation/06_rag_assistant.py
```

---

# 9. Reproducibility

The repository is organized task-by-task to make the experimental workflow easier to inspect and reproduce.

For reproducible execution:

- keep all dataset files under `datasets/`;
- keep location-recommendation data under `datasets/location-based/`;
- keep route-generation data under `datasets/route-based/`;
- use relative paths instead of machine-specific absolute paths;
- keep model files under `location-recommendation-system/models/`;
- do not commit API keys or local environment files;
- keep generated cache files such as `__pycache__/` outside version control;
- document external LLM-based or manually conducted experiments separately when they cannot be reproduced entirely from the local Python code.

