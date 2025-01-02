# AI/ML Learning Repository

This repository documents my journey learning artificial intelligence and machine learning, following a structured curriculum that progresses from traditional AI through modern deep learning and generative AI systems.

## Overview

```mermaid
flowchart LR
    %% Style definitions with enhanced contrast
    classDef aiClass fill:#E6F3FF,stroke:#2B6CB0,stroke-width:2px,color:#1A365D
    classDef mlClass fill:#E6FFE6,stroke:#2F855A,stroke-width:2px,color:#1C4532
    classDef dlClass fill:#C6F6D5,stroke:#276749,stroke-width:2px,color:#1C4532
    classDef genClass fill:#FED7E2,stroke:#97266D,stroke-width:2px,color:#521B41
    classDef traditionalClass fill:#F7FAFC,stroke:#4A5568,stroke-width:2px,color:#1A202C
    classDef applicationClass fill:#FFFFFF,stroke:#4A5568,stroke-width:1px,color:#1A202C
    classDef focusClass stroke:#FF0000,stroke-width:4px

    %% Main AI node
    AI[("Artificial Intelligence 🧠")]

    %% Primary branches
    TRAD["Traditional AI"]
    ML["Machine Learning ⚙️"]
    DL["Deep Learning 🔄"]
    GEN["Generative AI 💡"]

    %% Traditional AI components
    subgraph Traditional["Traditional Approaches"]
        direction TB
        TRAD1["Rule-Based Systems"]
        TRAD2["Expert Systems"]
        TRAD3["Symbolic AI"]
        TRAD4["Search Algorithms"]
        TRAD5["Knowledge Representation"]
    end

    %% Machine Learning components
    subgraph MLTypes["Machine Learning Types"]
        direction TB
        ML1["Supervised Learning<br>(Scikit-learn)"]
        ML2["Unsupervised Learning"]
        ML3["Reinforcement Learning"]
        ML4["Ensemble Learning"]
    end

    subgraph MLEval["Model Evaluation"]
        direction TB
        ME1["Performance Metrics<br>(Accuracy, Precision, Recall, F1)"]
        ME2["Bias-Variance Tradeoff"]
        ME3["Hyperparameter Tuning"]
    end

    %% Deep Learning components
    subgraph DLTypes["Deep Learning Architecture"]
        direction TB
        DL1["Convolutional Neural Networks"]
        DL2["Recurrent Neural Networks"]
        DL3["Transformer Networks"]
        DL4["GANs"]
    end

    subgraph DLComp["Deep Learning Components"]
        direction TB
        DC1["Activation Functions<br>(ReLU, Sigmoid, Tanh)"]
        DC2["Loss Functions<br>(MSE, Cross-Entropy)"]
        DC3["Optimization Algorithms<br>(SGD, Adam, RMSprop)"]
    end

    %% Generative AI components
    subgraph GENTypes["Generative Models"]
        direction TB
        GEN1["Large Language Models<br>(PyTorch, Transformers)"]
        GEN2["Diffusion Models"]
        GEN3["VAEs"]
    end

    subgraph GenComp["Generative AI Components"]
        direction TB
        GC1["Prompt Engineering"]
        GC2["Fine-tuning"]
        GC3["Evaluation Metrics"]
    end

    %% MLOps and Leadership
    subgraph MLOps["MLOps & Infrastructure"]
        direction TB
        MO1["Docker/Kubernetes"]
        MO2["Cloud Platforms<br>(AWS SageMaker, GCP, Azure)"]
        MO3["Model Serving<br>(REST APIs, TF Serving, TorchServe)"]
        MO4["Feature Stores<br>(Feast)"]
        MO5["Workflow Orchestration<br>(Airflow, Kubeflow)"]
        MO6["Model Versioning/Tracking<br>(MLflow, DVC)"]
        MO7["Monitoring & Observability"]
    end
    
    subgraph Leadership["Leadership & Specialization"]
        direction TB
        LS1["Team Management & Mentorship"]
        LS2["Technical Strategy & Vision"]
        LS3["Communication & Collaboration"]
        LS4["MLOps & LLMs"]
    end

    %% Core relationships
    AI --> TRAD & ML
    ML --> DL
    DL --> GEN

    %% Component relationships
    TRAD --> Traditional
    ML --> MLTypes
    ML --> MLEval
    DL --> DLTypes
    DL --> DLComp
    GEN --> GENTypes
    GEN --> GenComp

    %% Cross-component relationships
    DL3 --> GEN1
    ML3 -.-> DL2
    TRAD3 -.-> ML1
    DLTypes --> DL4

    %% Examples
    ML_Ex[/"Examples:
    - Spam filters
    - Recommendation systems"/]
    DL_Ex[/"Examples:
    - Image recognition
    - Self-driving cars
    - Speech recognition"/]
    GEN_Ex[/"Examples:
    - ChatGPT
    - DALL-E 2
    - Bard
    - Code generation"/]

    MLTypes -.-> ML_Ex
    DLTypes -.-> DL_Ex
    GENTypes -.-> GEN_Ex

    %% Apply styles
    class AI aiClass
    class ML,ML1,ML2,ML3,ML4,ME1,ME2,ME3 mlClass
    class DL,DL1,DL2,DL3,DL4,DC1,DC2,DC3 dlClass
    class GEN,GEN1,GEN2,GEN3,GC1,GC2,GC3 genClass
    class TRAD,TRAD1,TRAD2,TRAD3,TRAD4,TRAD5 traditionalClass
    class ML_Ex,DL_Ex,GEN_Ex applicationClass
    class MLOps,Leadership focusClass

    %% Connect to MLOps and Leadership
    ML --> MLOps
    DL --> MLOps
    GEN --> MLOps
    ML --> Leadership
    DL --> Leadership
    GEN --> Leadership
```

This diagram illustrates the relationships between different branches of AI, from traditional approaches through modern deep learning and generative AI systems.

## Repository Structure

```
ai-ml-roadmap/
├── data/                    # Dataset storage and processing utilities
├── docs/                    # Documentation and learning materials
│   ├── diagrams/             # Visual explanations of concepts
│   └── notes/                # In-depth exploration of ML topics
│       ├── supervised.md      # Complete guide to supervised learning
│       ├── unsupervised.md    # Understanding unsupervised approaches
│       ├── types.md           # Overview of ML categories
│       └── reinforcement.md   # Guide to reinforcement learning
├── blogs/                   # Blog posts and learning reflections
│   └── humsman_vs_machine_learning.md # My first blog post on parallel between humans and machines learning
├── frameworks/              # Implementations using specific AI/ML frameworks
│   └── langchain/            # LangChain: Projects or experiments utilizing the LangChain framework 
│                             # for building applications with large language models
│   └── <future-framework>/   # Future implementations (TensorFlow, PyTorch, Scikit-learn)
├── notebooks/               # Hands-on learning, depth of concepts, and implementation
│   ├── algorithms/           # Algorithm implementations
│   │   └── [supervised_readme.md](notebooks/algorithms/supervised_readme.md)  # Detailed overview of supervised learning
│   ├── deep_learning/        # Neural network concepts and implementations
│   ├── generative_ai/        # Generative AI experiments and models
│   ├── machine_learning_basics/  # Foundation ML concepts
│   │   ├── supervised/            # Understanding supervised learning approaches
│   │   ├── unsupervised/          # Exploring unsupervised techniques
│   │   └── reinforcement/         # Introduction to reinforcement learning
│   ├── mlops/                # MLOps practices and infrastructure
│   └── traditional_ai/       # Classical AI approaches (rule-based systems, search algorithms, knowledge representation)
└── projects/                 # Real-world applications
    ├── generative_ai/         # Generative AI projects and experiments
    ├── supervised/            # Supervised learning projects
    │   ├── plant_classification/  # Classify plant species using supervised learning
    │   └── text_classification/   # A complete text classification system
    │       ├── api_service/       # Backend API implementation
    │       ├── frontend/          # User interface and visualization
    │       └── ml_service/        # Core machine learning service
    └── unsupervised/         # Unsupervised learning projects

```

## Core Learning Areas

### Traditional AI
- Rule-Based Systems
- Expert Systems
- Symbolic AI
- Search Algorithms
- Knowledge Representation

### Machine Learning
- **Supervised Learning** (using Scikit-learn): Classification, regression, decision trees
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Reinforcement Learning**: Policy optimization, Q-learning
- **Model Evaluation**: Performance metrics, bias-variance tradeoff, hyperparameter tuning

### Deep Learning
- Neural Networks and Advanced Architectures
- Activation Functions (ReLU, Sigmoid, Tanh)
- Loss Functions (MSE, Cross-Entropy)
- Optimization Algorithms (SGD, Adam, RMSprop)

### MLOps & Infrastructure
- Docker/Kubernetes for containerization
- Cloud Platforms (AWS SageMaker, GCP, Azure)
- Model Serving (REST APIs, TF Serving, TorchServe)
- Feature Stores (Feast)
- Workflow Orchestration (Airflow, Kubeflow)
- Model Versioning/Tracking (MLflow, DVC)
- Monitoring & Observability

## Key Projects
- `projects/supervised/text_classification/`: Complete text classification system with API and frontend
- `projects/supervised/plant_classification/`: Plant species classification using supervised learning
- Additional projects in development

## Roadmap

### Current Focus
- Implementing core supervised learning algorithms
- Building out the plant classification system
- Developing the text classification service
- MLOps infrastructure setup

### Next Steps
- Expand deep learning implementations
- Add generative AI experiments
- Enhance MLOps practices
- Develop leadership and mentorship skills

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-ml-roadmap.git
   cd ai-ml-roadmap
   ```

2. Review the learning materials:
   - Start with [`notebooks/algorithms/supervised_readme.md`](notebooks/algorithms/supervised_readme.md) for fundamentals
   - Explore `projects` for practical implementations
   - Check `docs` for detailed explanations and diagrams

3. Follow project-specific setup instructions in their respective directories

## Contributing

This is a personal learning repository but suggestions and discussions are welcome through issues and discussions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
