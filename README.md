# data-migration-poc

Functionality Explanation
The application automates the process of mapping and migrating data from multiple source schemas to a unified target schema. Here's a high-level overview of its functionality:
Input Handling:
Users upload up to three source CSV files containing data and a target schema CSV file (containing only headers, no data).

The application reads the column names (schemas) from all files and, for source files, their data.

Schema Comparison:
Uses the SentenceTransformer model (all-MiniLM-L6-v2) to generate embeddings for column names.

Computes a cosine similarity matrix to compare source and target field names semantically (e.g., "CustomerName" might match "ClientName" with a high similarity score).

Field Mapping:
For each source field, identifies the most similar target field based on the highest similarity score.

If a source field's similarity score is below a threshold (default 0.5), it is suggested as a new field to add to the target schema.

Schema Update:
Updates the target schema by adding any source fields that don't have a good match (based on the threshold).

Recalculates the similarity matrix for the updated target schema.

Data Migration:
Maps data from source files to the target schema based on the field mappings.

Only maps data for fields with similarity scores above the threshold.

Creates a unified DataFrame with the target schema, filling unmapped columns with None.

Output:
Displays the initial and updated similarity matrices in the UI.

Displays the updated target schema and a preview of the unified data.

Saves the unified data to a CSV file (populated_target_schema.csv).

Key Features
Semantic Mapping: Uses NLP to match fields based on meaning, not just exact names.

Threshold-Based Filtering: Allows control over which fields are mapped or suggested as new based on a similarity threshold.

Data Unification: Combines data from multiple sources into a single target schema.

Error Handling: Warns users about empty or invalid CSV files.

Interactive UI: Streamlit provides a user-friendly interface for uploading files and viewing results.

Potential Use Cases
Data Integration: Combining data from multiple sources (e.g., different databases or applications) into a unified schema.

Data Migration: Migrating data to a new system with a different schema.

Schema Evolution: Suggesting new fields for a target schema based on source data.

Limitations
Fixed Number of Source Files: The UI only allows up to three source files.

Threshold Sensitivity: The similarity threshold (0.5) may need tuning for different datasets.

No Data Validation: The code doesn't validate the content of the source data (e.g., data types or consistency).

Single Model: Relies on all-MiniLM-L6-v2 for embeddings, which may not handle all cases perfectly.

