# imdb_recommender
Content based filtering recommender for films in the IMDB dataset that leverages the use of LLMs to extract latent features from text.
.
├── utils/                  # Utility modules for core functionality
│   ├── processor.py        # Contains the RecProcessor class for data cleaning and feature extraction
│   └── recommender.py      # Contains the Recommender class to compute and rank movie recommendations
│
├── TMDb_updated.CSV        # The movie dataset (Top 10,000 TMDb titles)
├── main.py                 # Main script to run preprocessing and generate recommendations
├── requirements.txt        # List of required libraries for pip installation
└── README.md               # Project overview and usage instructions

