# STEM Connect AI

### Connecting Scientific Knowledge to Real-World Engineering Projects

STEM Connect AI is an AI-powered writing assistant designed specifically
for STEM high school students. It helps students connect scientific
concepts and Learning Outcomes (LOs) from their academic subjects
to their Capstone engineering projects.

The platform allows students to upload their own scientific subject
PDF and Capstone project PDF, select a Learning Outcome, and generate
a 200–250 word connection essay. Students can also submit their own
essays and receive formative feedback based on the Capstone Journal Rubric.

The goal is to help students build meaningful connections between
what they learn in the classroom and what they develop through
engineering and project-based learning.

---

## The Problem

STEM students study scientific subjects such as Chemistry, Physics,
Biology, and Mathematics while simultaneously working on interdisciplinary
Capstone engineering projects.

However, connecting the concepts learned in these subjects to the
requirements, design decisions, and experiments of a Capstone project
can be challenging.

Students may understand a scientific concept but struggle to explain
how it applies to their engineering project.

They also need to write structured connection essays and evaluate
their work according to the Capstone Journal Rubric.

This creates several challenges:

- Identifying relevant scientific concepts in curriculum documents.
- Connecting Learning Outcomes to real engineering requirements.
- Finding evidence in Capstone project documentation.
- Writing focused scientific connection essays.
- Understanding how an essay performs against the assessment rubric.

STEM Connect AI aims to make this process more structured,
accessible, and evidence-oriented.

---

## The Solution

STEM Connect AI provides a document-based workflow that helps students
analyze their own curriculum and Capstone project documents.

Instead of relying entirely on generic AI knowledge, the application
uses the documents uploaded by the student as the primary context
for its analysis.

Students can choose between two workflows:

### 1. Generate a Connection Essay

Students upload:

- A scientific subject PDF containing curriculum information
  and Learning Outcomes.
- A Capstone project PDF containing project requirements,
  design choices, and other relevant information.

They then enter the Learning Outcome code they want to connect
to their Capstone project.

The application:

1. Extracts text from both PDF documents.
2. Identifies the requested Learning Outcome in the curriculum.
3. Extracts relevant scientific concepts when available.
4. Retrieves potentially relevant Capstone evidence using
   TF-IDF-based text similarity.
5. Uses a Gemini model to analyze the scientific connection.
6. Generates a 200–250 word connection essay.
7. Displays the essay, word count, supporting evidence,
   potential limitations, and personalization notes.
8. Allows the student to download the essay as a text file.

The generated essay is intended as a draft that students should
review, verify, and personalize using their actual project experience.

### 2. Evaluate My Own Essay

Students can also submit an essay they have already written.

The application evaluates the essay using four criteria
from the Capstone Journal Rubric:

- Learning Transfer
- Team Collaboration
- Using the Engineering Design Process
- Personal Reflection

For each criterion, the application provides:

- An estimated rubric level.
- Evidence from the submitted essay.
- An explanation of the assessment.
- Suggestions for improvement.
- A reflective question to help the student improve their writing.

The application also provides an overall estimated level
and allows students to download their feedback.

The results are formative feedback, not an official school grade.

---

## Key Features

- **Student-Provided Documents:** Students upload their own
  scientific subject and Capstone PDFs.

- **Learning Outcome Selection:** Students specify the exact
  LO code they want to explore.

- **Scientific Concept Analysis:** The application analyzes
  available scientific concepts and their potential relevance
  to the Capstone project.

- **Evidence-Oriented Retrieval:** TF-IDF similarity helps
  identify potentially relevant passages from the Capstone PDF.

- **AI-Powered Essay Generation:** Gemini generates a focused
  connection essay with a target length of 200–250 words.

- **Rubric-Based Feedback:** Students receive feedback across
  four Capstone Journal Rubric criteria.

- **Evidence and Limitation Awareness:** The application
  attempts to distinguish supported connections from limited
  or unsupported ones.

- **Essay Export:** Generated essays and evaluation reports
  can be downloaded as text files.

- **Interactive Web Interface:** Streamlit provides an interface
  for document uploads, workflow selection, generation, and evaluation.

---

## How It Works

```text
                  Student
                     |
                     v
          Upload Two PDF Documents
                     |
          +----------+----------+
          |                     |
          v                     v
   Scientific Subject      Capstone Project
          |                     |
          v                     v
   Extract Curriculum      Extract Project Text
   and Learning Outcomes        |
          |                     v
          |              TF-IDF Retrieval
          |                     |
          +----------+----------+
                     |
                     v
             Select Learning Outcome
                     |
                     v
          Analyze Scientific Connection
                     |
                     v
              Gemini AI Model
                     |
          +----------+----------+
          |                     |
          v                     v
   Generate Connection     Evaluate Student's
        Essay                   Essay
          |                     |
          v                     v
   200–250 Word Draft      Rubric-Based Feedback
          |                     |
          +----------+----------+
                     |
                     v
              Student Review
```

---

## Technology Stack

### Programming Language

**Python**

Used for PDF processing, document retrieval, AI integration,
data validation, and application logic.

### Web Application

**Streamlit**

Provides the interactive web interface, including PDF uploaders,
input fields, workflow selection, essay display, and downloadable results.

### Large Language Model

**Google Gemini API**

Used to analyze the supplied curriculum and Capstone evidence,
generate connection essays, and provide rubric-based writing feedback.

The application allows the user to configure the API key and model
through the interface.

### LLM Integration

**LangChain**

Used to construct prompts and organize interactions with the Gemini
model. Structured output is used to represent analysis results,
essay drafts, and rubric feedback.

### PDF Processing

**PyMuPDF**

Extracts text from uploaded PDF documents and preserves page-level
information for the retrieval workflow.

### Information Retrieval

**Scikit-learn**

TF-IDF vectorization and cosine similarity are used to identify
potentially relevant Capstone document pages.

This is a lightweight lexical retrieval approach, not a full
semantic vector database or embedding-based retrieval system.

### Data Validation

**Pydantic**

Defines structured output models for scientific concept analysis,
essay generation, and rubric-based feedback.

---

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- A Google Gemini API key.
- The dependencies listed in `requirements.txt`.

### 1. Clone the Repository

```bash
git clone https://github.com/mohammedwetwet/stem-connection-ai.git
```

Replace `mohammedwetwet` with your GitHub username.

Navigate to the project directory:

```bash
cd stem-connect-ai
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal.

Open that URL in your browser to use the application.

### 5. Configure Gemini

Enter your Google Gemini API key in the sidebar.

Select a supported Gemini model and upload your documents.

Keep your API key private. Never share it with anyone or commit
it to your repository.

---

## Example Workflow

Imagine a student working on a Capstone project involving
passive food preservation packaging.

The student wants to connect a Chemistry Learning Outcome
about chemical kinetics to the project.

The workflow would be:

1. Upload the Chemistry curriculum PDF.
2. Upload the Capstone project PDF.
3. Enter the relevant Chemistry Learning Outcome code.
4. Select "Generate a connection essay".
5. Review the generated essay and its supporting project evidence.
6. Verify scientific claims and personalize the draft.
7. Download the essay for further editing.

Alternatively, the student can write the essay independently,
select "Evaluate my own essay", and receive feedback based on
the Capstone Journal Rubric.

This is an illustrative example. The application should use the
actual documents supplied by the student rather than assuming
that every Capstone project involves food preservation.

---

## Project Structure

```text
stem-connect-ai/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

### File Descriptions

- `app.py`: Main Streamlit application, PDF processing,
  evidence retrieval, Gemini integration, essay generation,
  and rubric evaluation.

- `requirements.txt`: Python dependencies required to run
  the application.

- `README.md`: Project documentation and setup instructions.

- `.gitignore`: Excludes virtual environments, local secrets,
  caches, and other files that should not be committed.

---

## Limitations

STEM Connect AI is a prototype intended to support student learning
and writing.

Its current implementation has several limitations:

- PDF text extraction may not work correctly with scanned documents
  that require Optical Character Recognition (OCR).

- Learning Outcome identification depends on the text and formatting
  of the uploaded curriculum document.

- TF-IDF retrieval identifies lexical similarities and may miss
  relevant connections expressed using different terminology.

- AI-generated connections and essays may contain inaccuracies
  and must be checked against the original documents.

- The rubric evaluation is an AI-generated estimate and may differ
  from an instructor's assessment.

- The current application does not guarantee that every scientific
  connection is valid or sufficiently supported by project evidence.

Future improvements could include stronger semantic retrieval,
better support for different curriculum formats, more reliable
source citations, and additional evaluation tests.

---

## Responsible Use

STEM Connect AI is designed to support learning, scientific reasoning,
and reflective writing.

Students should:

- Verify scientific claims against reliable sources.
- Check whether the proposed connection is supported by their project.
- Avoid presenting hypothetical results as completed experiments.
- Personalize reflections using their actual experiences.
- Follow their school's rules regarding AI-assisted writing.
- Protect API keys and confidential project information.

The generated essay should be treated as a draft for learning
and revision, not as a substitute for the student's own thinking.

---

## Future Improvements

Potential directions for future development include:

- Semantic search and embedding-based retrieval.
- More robust Learning Outcome extraction.
- Improved source attribution for generated claims.
- OCR support for scanned curriculum documents.
- More detailed rubric calibration and evaluation.
- Automated checks for essay length and evidence coverage.
- Support for additional STEM subjects and curriculum formats.
- Testing with real student essays and educator feedback.

These are potential improvements, not claims about features
already implemented.

---

## Author
Mohamed Adel Yousef Wetwet - STEM Student S'28
Developed as an AI-powered educational project focused on helping
STEM students connect scientific knowledge with engineering practice.

## License

A license has not yet been selected.

Choose an appropriate license before distributing the project
for reuse or modification.
