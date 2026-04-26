"""
Curated skill vocabulary for extracting structured skills from free-text job
descriptions. Organized into broad categories so the graph layer can reason
about adjacency between skills (e.g. Python → Pandas → scikit-learn).

Each skill has:
  - canonical name (the form we display and store)
  - a list of surface aliases that appear in real job postings (case-insensitive)
  - a category tag

The vocabulary is deliberately tech-leaning because the proposal scopes the
project to technical career paths. Extend `SKILLS` to broaden coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def all_surface_forms(self) -> tuple[str, ...]:
        return (self.name,) + self.aliases


SKILLS: tuple[Skill, ...] = (
    # Programming languages
    Skill("Python", "language", ("python3", "py")),
    Skill("Java", "language", ()),
    Skill("JavaScript", "language", ("js", "ecmascript")),
    Skill("TypeScript", "language", ("ts",)),
    Skill("C++", "language", ("cpp", "c plus plus")),
    Skill("C#", "language", ("c sharp", "csharp", "dotnet")),
    Skill("Go", "language", ("golang",)),
    Skill("Rust", "language", ()),
    Skill("Ruby", "language", ()),
    Skill("PHP", "language", ()),
    Skill("Swift", "language", ()),
    Skill("Kotlin", "language", ()),
    Skill("Scala", "language", ()),
    Skill("R", "language", ()),
    Skill("MATLAB", "language", ("matlab",)),
    Skill("SQL", "language", ()),
    Skill("Bash", "language", ("shell scripting", "shell script")),
    # Web frameworks / frontend
    Skill("React", "web", ("reactjs", "react.js")),
    Skill("Angular", "web", ("angularjs",)),
    Skill("Vue", "web", ("vuejs", "vue.js")),
    Skill("Next.js", "web", ("nextjs",)),
    Skill("Node.js", "web", ("nodejs", "node js")),
    Skill("Django", "web", ()),
    Skill("Flask", "web", ()),
    Skill("FastAPI", "web", ()),
    Skill("Spring", "web", ("spring boot", "springboot")),
    Skill("Express", "web", ("express.js", "expressjs")),
    Skill("HTML", "web", ("html5",)),
    Skill("CSS", "web", ("css3",)),
    Skill("Tailwind", "web", ("tailwindcss",)),
    # Data & ML
    Skill("Pandas", "data", ()),
    Skill("NumPy", "data", ("numpy",)),
    Skill("scikit-learn", "ml", ("sklearn", "scikit learn")),
    Skill("TensorFlow", "ml", ()),
    Skill("PyTorch", "ml", ()),
    Skill("Keras", "ml", ()),
    Skill("XGBoost", "ml", ()),
    Skill("Hugging Face", "ml", ("huggingface",)),
    Skill("LangChain", "ml", ()),
    Skill("Spark", "data", ("apache spark", "pyspark")),
    Skill("Hadoop", "data", ()),
    Skill("Kafka", "data", ("apache kafka",)),
    Skill("Airflow", "data", ("apache airflow",)),
    Skill("dbt", "data", ()),
    Skill("Snowflake", "data", ()),
    Skill("BigQuery", "data", ("google bigquery",)),
    Skill("Redshift", "data", ("amazon redshift",)),
    Skill("Tableau", "data", ()),
    Skill("Power BI", "data", ("powerbi",)),
    Skill("Looker", "data", ()),
    Skill("Excel", "data", ("microsoft excel",)),
    # ML concepts (broad)
    Skill("Machine Learning", "ml", ("ml",)),
    Skill("Deep Learning", "ml", ("dl",)),
    Skill("NLP", "ml", ("natural language processing",)),
    Skill("Computer Vision", "ml", ("cv",)),
    Skill("Reinforcement Learning", "ml", ("rl",)),
    Skill("Statistics", "ml", ("statistical analysis",)),
    Skill("Data Modeling", "data", ()),
    Skill("ETL", "data", ("extract transform load",)),
    # Databases
    Skill("PostgreSQL", "database", ("postgres",)),
    Skill("MySQL", "database", ()),
    Skill("MongoDB", "database", ("mongo",)),
    Skill("Redis", "database", ()),
    Skill("Elasticsearch", "database", ("elastic search",)),
    Skill("DynamoDB", "database", ()),
    Skill("Cassandra", "database", ()),
    Skill("SQLite", "database", ()),
    Skill("Oracle", "database", ("oracle db",)),
    # Cloud & DevOps
    Skill("AWS", "cloud", ("amazon web services",)),
    Skill("Azure", "cloud", ("microsoft azure",)),
    Skill("GCP", "cloud", ("google cloud platform", "google cloud")),
    Skill("Docker", "devops", ()),
    Skill("Kubernetes", "devops", ("k8s",)),
    Skill("Terraform", "devops", ()),
    Skill("Ansible", "devops", ()),
    Skill("Jenkins", "devops", ()),
    Skill("GitHub Actions", "devops", ("github action",)),
    Skill("GitLab CI", "devops", ("gitlab ci/cd",)),
    Skill("CI/CD", "devops", ("ci cd", "continuous integration")),
    Skill("Linux", "devops", ()),
    Skill("Git", "devops", ()),
    # Mobile
    Skill("Android", "mobile", ()),
    Skill("iOS", "mobile", ()),
    Skill("React Native", "mobile", ()),
    Skill("Flutter", "mobile", ()),
    # Testing & quality
    Skill("Unit Testing", "testing", ("unit tests",)),
    Skill("Selenium", "testing", ()),
    Skill("Cypress", "testing", ()),
    Skill("Jest", "testing", ()),
    Skill("Pytest", "testing", ()),
    Skill("JUnit", "testing", ()),
    # Design & methodology
    Skill("Agile", "methodology", ("agile methodology",)),
    Skill("Scrum", "methodology", ()),
    Skill("Kanban", "methodology", ()),
    Skill("Jira", "tooling", ()),
    Skill("Confluence", "tooling", ()),
    # Security
    Skill("Penetration Testing", "security", ("pen testing", "pentest")),
    Skill("OWASP", "security", ()),
    Skill("Cryptography", "security", ()),
    Skill("Network Security", "security", ()),
    # Business / soft tech
    Skill("REST API", "web", ("restful api", "rest apis", "rest")),
    Skill("GraphQL", "web", ()),
    Skill("Microservices", "architecture", ()),
    Skill("System Design", "architecture", ()),
    Skill("Distributed Systems", "architecture", ()),
    Skill("OOP", "architecture", ("object oriented programming",)),
    Skill("Functional Programming", "architecture", ()),
    Skill("Data Structures", "fundamentals", ()),
    Skill("Algorithms", "fundamentals", ()),
    # Domain-adjacent (non-tech roles still use these)
    Skill("Project Management", "business", ("project mgmt",)),
    Skill("Product Management", "business", ("product mgmt",)),
    Skill("Stakeholder Management", "business", ()),
    Skill("Business Analysis", "business", ()),
    Skill("Data Analysis", "business", ()),
    Skill("Salesforce", "business", ()),
    Skill("SAP", "business", ()),
    Skill("Adobe Photoshop", "design", ("photoshop",)),
    Skill("Adobe Illustrator", "design", ("illustrator",)),
    Skill("Figma", "design", ()),
    Skill("UX Design", "design", ("ux", "user experience")),
    Skill("UI Design", "design", ("ui",)),
)


def build_alias_index() -> dict[str, str]:
    """Map every lowercase surface form to its canonical skill name."""
    index: dict[str, str] = {}
    for skill in SKILLS:
        for form in skill.all_surface_forms():
            index[form.lower()] = skill.name
    return index


def canonical_names() -> list[str]:
    return [s.name for s in SKILLS]


def category_of(name: str) -> str | None:
    for s in SKILLS:
        if s.name == name:
            return s.category
    return None
