"""Domain ontology for the Senior AI Engineer JD.

This module encodes the *meaning* of the role as weighted concept clusters and
several lookup taxonomies (titles, companies, locations). It is the bridge
between "what the JD says" and "what the JD means" — e.g. a candidate who wrote
"built a recommendation system that served personalised results to millions"
matches the ranking/retrieval clusters without ever saying "RAG" or "Pinecone".

Everything here is hand-authored from a close reading of job_description.docx.
This is the part of the system that does "deep job understanding"; it is
compiled once, offline, and needs no LLM at ranking time.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Concept clusters
# ---------------------------------------------------------------------------
# Each cluster maps to a JD requirement. `weight` reflects how decisive the
# cluster is for *this* role (must-haves dominate). `terms` are normalised
# phrases (1-3 words, see text.normalize) that signal the concept; they are
# matched against candidate evidence text, NOT just the skills array.
#
# `kind`:
#   core      -> a JD "absolutely need" must-have
#   plus      -> a JD "would like to have" nice-to-have
#   support   -> general engineering / production signal that reinforces fit
#   negative  -> signals the JD explicitly does NOT want (handled as penalties)

CONCEPTS: dict[str, dict] = {
    # ---- MUST-HAVES -------------------------------------------------------
    "embeddings_retrieval": {
        "kind": "core",
        "weight": 1.00,
        "terms": [
            "embedding", "embeddings", "sentence transformers", "sentence transformer",
            "dense retrieval", "dense vector", "semantic search", "semantic retrieval",
            "bi encoder", "cross encoder", "text encoder", "text encoders",
            "vector representation", "vector representations", "bge", "e5",
            "embedding model", "embedding models", "neural retrieval", "nearest neighbor",
            "nearest neighbour", "ann", "retrieval system", "retrieval systems",
        ],
    },
    "vector_db_hybrid": {
        "kind": "core",
        "weight": 1.00,
        "terms": [
            "vector database", "vector db", "vector search", "vector store",
            "pinecone", "weaviate", "qdrant", "milvus", "faiss", "pgvector",
            "opensearch", "elasticsearch", "elastic search", "hybrid search",
            "hybrid retrieval", "bm25", "inverted index", "search index",
            "indexing", "search infrastructure", "search backend", "ann index",
            "haystack",
        ],
    },
    "ranking_reco": {
        "kind": "core",
        "weight": 1.00,
        "terms": [
            "ranking", "ranker", "re ranking", "reranking", "re rank", "rerank",
            "learning to rank", "ltr", "relevance", "relevance tuning",
            "recommendation", "recommendation system", "recommendation systems",
            "recommender", "recommender system", "personalization", "personalisation",
            "discovery feed", "search ranking", "content matching", "matching system",
            "ranking model", "ranking models", "ranking layer", "ranking systems",
            "candidate ranking", "search and discovery", "search relevance",
            # plain-language paraphrases used by "Tier 5" hidden gems who avoid
            # buzzwords (e.g. "decide what to show", "connect them to the most
            # relevant matches", "surface relevant content"):
            "surface relevant", "most relevant", "relevant matches",
            "relevant results", "what to show", "connect users", "connect them",
            "personalized results", "personalised results", "what users see",
        ],
    },
    "evaluation": {
        "kind": "core",
        "weight": 0.95,
        "terms": [
            "ndcg", "mrr", "map", "mean average precision", "precision recall",
            "precision at", "recall at", "a b test", "a b testing", "ab test",
            "ab testing", "offline evaluation", "offline metrics", "online metrics",
            "evaluation framework", "eval framework", "eval harness", "ranking metrics",
            "retrieval quality", "offline to online", "experimentation", "experiment",
            "holdout", "benchmark", "benchmarks", "evaluation rigor", "evaluation rigour",
            "golden set", "relevance judgments", "relevance judgements",
        ],
    },
    # ---- NICE-TO-HAVES ----------------------------------------------------
    "llm_finetune": {
        "kind": "plus",
        "weight": 0.55,
        "terms": [
            "lora", "qlora", "peft", "fine tuning", "fine tune", "fine tuned",
            "instruction tuning", "supervised fine", "rlhf", "dpo", "model adaptation",
            "distillation", "quantization", "quantisation",
        ],
    },
    "llm_apps": {
        "kind": "plus",
        "weight": 0.40,
        "terms": [
            "llm", "llms", "large language model", "large language models",
            "rag", "retrieval augmented", "retrieval augmented generation",
            "prompt engineering", "llamaindex", "langchain", "hugging face",
            "huggingface", "transformers", "gpt", "agentic",
        ],
    },
    "ltr_models": {
        "kind": "plus",
        "weight": 0.45,
        "terms": [
            "xgboost", "lightgbm", "gradient boosting", "gradient boosted",
            "learning to rank", "neural ranking", "two tower", "two-tower",
            "collaborative filtering", "matrix factorization", "matrix factorisation",
        ],
    },
    "mlops": {
        "kind": "support",
        "weight": 0.40,
        "terms": [
            "mlops", "mlflow", "kubeflow", "bentoml", "feature store",
            "model serving", "model deployment", "experiment tracking",
            "weights biases", "weights and biases", "model monitoring",
            "feature pipeline", "feature pipelines", "training pipeline",
            "inference pipeline", "vertex ai", "sagemaker",
        ],
    },
    "nlp_ir": {
        "kind": "support",
        "weight": 0.55,
        "terms": [
            "nlp", "natural language processing", "information retrieval",
            "text classification", "named entity", "question answering",
            "document classification", "sentiment analysis", "topic modeling",
            "topic modelling", "language model", "tokenization", "tokenisation",
            "search query", "query understanding",
        ],
    },
    "production_scale": {
        "kind": "support",
        "weight": 0.50,
        "terms": [
            "production", "in production", "deployed", "shipped", "shipping",
            "real users", "at scale", "millions", "billions", "low latency",
            "latency", "throughput", "serving", "live a b", "live ab",
            "rolled out", "went live", "high traffic", "real time", "realtime",
            "distributed systems", "large scale", "scalable",
        ],
    },
    "python_eng": {
        "kind": "support",
        "weight": 0.35,
        "terms": [
            "python", "pytorch", "tensorflow", "scikit learn", "sklearn",
            "numpy", "pandas", "spark", "pyspark", "airflow", "api design",
            "microservice", "microservices", "system design", "code quality",
            "data pipeline", "data pipelines",
        ],
    },
    # ---- NEGATIVE / ANTI-FIT (used to compute penalties) ------------------
    "cv_speech": {
        "kind": "negative",
        "weight": 0.0,
        "terms": [
            "computer vision", "opencv", "yolo", "image classification",
            "object detection", "image segmentation", "convolutional",
            "speech recognition", "asr", "text to speech", "tts", "voice",
            "diffusion model", "diffusion models", "image generation", "gans",
            "gan", "facial recognition", "ocr", "video analytics", "pose estimation",
        ],
    },
    "non_tech": {
        "kind": "negative",
        "weight": 0.0,
        "terms": [
            "marketing", "brand", "branding", "campaign", "campaigns", "seo",
            "social media", "sales", "salesforce", "lead generation", "crm",
            "accounting", "bookkeeping", "tally", "invoice", "payroll", "audit",
            "recruiting", "recruitment", "talent acquisition", "onboarding",
            "human resources", "employee", "logistics", "warehouse", "fulfillment",
            "fulfilment", "supply chain", "procurement", "graphic design",
            "visual design", "packaging", "typography", "illustrator", "photoshop",
            "copywriting", "content writing", "blog", "civil engineering",
            "structural", "mechanical design", "cad", "hvac", "manufacturing process",
            "operations management", "customer support", "ticket", "six sigma",
        ],
    },
    "research_only": {
        "kind": "negative",
        "weight": 0.0,
        "terms": [
            "phd", "doctoral", "postdoc", "publication", "published", "research paper",
            "peer reviewed", "thesis", "dissertation", "academic", "novel architecture",
            "state of the art", "neurips", "icml", "acl", "cvpr", "research lab",
        ],
    },
}

# Convenience groupings.
CORE_CLUSTERS = [k for k, v in CONCEPTS.items() if v["kind"] == "core"]
PLUS_CLUSTERS = [k for k, v in CONCEPTS.items() if v["kind"] == "plus"]
SUPPORT_CLUSTERS = [k for k, v in CONCEPTS.items() if v["kind"] == "support"]
POSITIVE_CLUSTERS = CORE_CLUSTERS + PLUS_CLUSTERS + SUPPORT_CLUSTERS
NEGATIVE_CLUSTERS = [k for k, v in CONCEPTS.items() if v["kind"] == "negative"]

# Human-readable labels used when generating reasoning strings.
CLUSTER_LABELS = {
    "embeddings_retrieval": "embeddings-based retrieval",
    "vector_db_hybrid": "vector search / hybrid retrieval infrastructure",
    "ranking_reco": "ranking & recommendation systems",
    "evaluation": "ranking evaluation (NDCG/MRR/A-B testing)",
    "llm_finetune": "LLM fine-tuning (LoRA/QLoRA/PEFT)",
    "llm_apps": "LLM / RAG application work",
    "ltr_models": "learning-to-rank models",
    "mlops": "ML production tooling",
    "nlp_ir": "NLP / information retrieval",
    "production_scale": "production ML at scale",
    "python_eng": "Python / ML engineering",
    "cv_speech": "computer-vision / speech",
    "non_tech": "non-technical work",
    "research_only": "pure-research signals",
}

# ---------------------------------------------------------------------------
# Title taxonomy  (priors only; evidence from descriptions can override)
# ---------------------------------------------------------------------------
# A title prior is a gentle nudge — the real domain signal comes from career
# descriptions. We never *gate* purely on title, because hidden gems carry weak
# titles and stuffers carry strong-sounding skills. But the current/most-recent
# title is genuine evidence and deserves weight.

TITLE_TIERS = {
    # Bullseye: exactly the role family the JD describes.
    "bullseye": [
        "senior ai engineer", "ai engineer", "lead ai engineer", "staff ai engineer",
        "machine learning engineer", "ml engineer", "senior machine learning",
        "staff machine learning", "lead machine learning", "applied ml engineer",
        "applied scientist", "senior applied scientist", "search engineer",
        "recommendation systems engineer", "relevance engineer", "nlp engineer",
        "senior nlp engineer",
    ],
    # Strong: ML/DS adjacent, very likely relevant.
    "strong": [
        "data scientist", "senior data scientist", "ai specialist", "ai research engineer",
        "research engineer", "senior software engineer (ml)", "ml scientist",
        "deep learning engineer",
    ],
    # Mid: technical SWE/data; relevant only if the career history shows ML.
    "mid": [
        "software engineer", "senior software engineer", "backend engineer",
        "full stack developer", "data engineer", "senior data engineer",
        "analytics engineer", "platform engineer", "staff software engineer",
    ],
    # Low: technical but distant from ML/IR; needs strong description evidence.
    "low": [
        "frontend engineer", "mobile developer", "devops engineer", "cloud engineer",
        "qa engineer", "java developer", ".net developer", "data analyst",
    ],
    # Off: non-technical roles. Genuine ones score ~0 on ML evidence; these are
    # the usual hosts for keyword-stuffer traps.
    "off": [
        "hr manager", "accountant", "mechanical engineer", "civil engineer",
        "marketing manager", "sales executive", "customer support",
        "operations manager", "project manager", "business analyst",
        "content writer", "graphic designer",
    ],
}
TITLE_PRIOR = {"bullseye": 1.0, "strong": 0.8, "mid": 0.5, "low": 0.32, "off": 0.12}


def title_tier(title: str) -> str:
    """Classify a job title into a tier (exact-ish match, longest first)."""
    t = title.lower().strip()
    # check the more specific tiers first; within a tier prefer exact then prefix
    for tier in ("bullseye", "strong", "mid", "low", "off"):
        for name in TITLE_TIERS[tier]:
            if t == name:
                return tier
    for tier in ("bullseye", "strong", "mid", "low", "off"):
        for name in TITLE_TIERS[tier]:
            if name in t or t in name:
                return tier
    return "mid"  # unknown technical-sounding title: neutral-ish, let evidence decide


# ---------------------------------------------------------------------------
# Company taxonomy
# ---------------------------------------------------------------------------
# The JD penalises careers spent *entirely* at IT-services/consulting firms,
# and values product-company experience. Company names in the dataset mix real
# Indian firms with placeholder names; we lean on the `industry` field too.

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "cognizant", "capgemini", "accenture",
    "tech mahindra", "mindtree", "hcl", "mphasis", "ltimindtree", "deloitte",
    "ibm", "dxc", "hexaware",
}
CONSULTING_INDUSTRIES = {"it services", "consulting"}

# Industries that indicate genuine product / tech companies.
PRODUCT_INDUSTRIES = {
    "software", "fintech", "e-commerce", "food delivery", "ai/ml", "saas",
    "adtech", "transportation", "insurance tech", "gaming", "healthtech",
    "healthtech ai", "conversational ai", "ai services", "edtech", "internet",
    "consumer electronics",
}
# Real Indian product companies that appear in the pool (positive signal even
# when the industry label is generic).
PRODUCT_COMPANIES = {
    "swiggy", "zomato", "flipkart", "cred", "razorpay", "meesho", "nykaa",
    "zoho", "freshworks", "ola", "inmobi", "vedantu", "phonepe", "paytm",
    "glance", "verloop.io", "haptik", "observe.ai", "locobuzz", "dream11",
    "rephrase.ai", "policybazaar", "aganitha", "swiggy", "myntra", "sharechat",
    "google", "apple", "adobe", "microsoft", "amazon", "uber", "netflix",
}


def is_consulting(company: str, industry: str) -> bool:
    c = (company or "").lower().strip()
    ind = (industry or "").lower().strip()
    if c in CONSULTING_FIRMS:
        return True
    # placeholder names (Hooli, Initech, ...) carry no consulting signal; rely on
    # industry only for the named consulting houses.
    return c in CONSULTING_FIRMS or (ind in CONSULTING_INDUSTRIES and c in CONSULTING_FIRMS)


def is_product(company: str, industry: str) -> bool:
    c = (company or "").lower().strip()
    ind = (industry or "").lower().strip()
    return c in PRODUCT_COMPANIES or ind in PRODUCT_INDUSTRIES


# ---------------------------------------------------------------------------
# Location taxonomy  (JD: Pune/Noida preferred; Tier-1 India welcome; outside
# India case-by-case with no visa sponsorship)
# ---------------------------------------------------------------------------
PREFERRED_CITIES = {"pune", "noida"}                       # JD's offices
TIER1_INDIA = {                                            # explicitly welcomed
    "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
    "noida", "pune", "bangalore", "bengaluru", "ncr", "delhi ncr",
}


def location_tier(location: str, country: str) -> str:
    loc = (location or "").lower()
    ctry = (country or "").lower()
    if ctry and ctry != "india":
        return "abroad"
    if any(city in loc for city in PREFERRED_CITIES):
        return "preferred"
    if any(city in loc for city in TIER1_INDIA):
        return "tier1"
    return "other_india"
