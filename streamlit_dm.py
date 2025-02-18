import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read schema and sample data from CSV (only header from target schema)


def read_schema_from_csv(file, is_target_schema=False):
    try:
        # If it's the target schema, we only need the header
        if is_target_schema:
            df = pd.read_csv(file, nrows=0)  # Read only headers (no data)
            return df.columns.tolist(), None
        else:
            df = pd.read_csv(file)
            if df.empty or df.columns.str.contains('Unnamed').all():
                st.warning(
                    f"Warning: The file {file.name} is empty or only contains headers.")
                return [], pd.DataFrame()
            else:
                st.write(f"Data from {file.name}:")
                st.dataframe(df.head())  # Show a preview of the data
                return df.columns.tolist(), df
    except pd.errors.EmptyDataError:
        st.warning(
            f"Warning: The file {file.name} is empty and cannot be processed.")
        return [], pd.DataFrame()

# Function to process the data and compute similarity matrix


def process_data(source_files, target_file, threshold=0.5):
    # Read source schemas
    source_data = [read_schema_from_csv(file) for file in source_files]
    source_schemas = [data[0] for data in source_data]

    # Combine all source fields
    source_fields = list(set(sum(source_schemas, [])))

    # Read target schema (only header, no data)
    target_schema, target_data = read_schema_from_csv(
        target_file, is_target_schema=True)
    if not target_schema:  # If the target schema is empty or not processed correctly
        st.warning("Target schema is empty, no processing can be done.")
        return None, None, None
    else:
        # Only show the target schema headers (no data displayed)
        st.write(f"Target Schema (Header Only): {target_schema}")

    # Load pre-trained embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Generate embeddings for source and target fields
    source_embeddings = model.encode(source_fields)
    target_embeddings = model.encode(target_schema)

    # Compute initial cosine similarity
    similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
    similarity_df = pd.DataFrame(
        similarity_matrix, index=source_fields, columns=target_schema)

    # Display similarity matrix
    st.write("Initial Similarity Matrix:")
    st.dataframe(similarity_df)

    # Find the best mapping for each source field
    field_mapping = {}
    for source_field in similarity_df.index:
        best_match = similarity_df.loc[source_field].idxmax()
        similarity_score = similarity_df.loc[source_field].max()
        field_mapping[source_field] = (best_match, similarity_score)

    # Define a similarity threshold for suggesting new fields
    new_fields = [field for field,
                  (_, score) in field_mapping.items() if score < threshold]

    # Add suggested new fields to the target schema
    updated_target_schema = target_schema + new_fields

    st.write("Updated Target Schema (with Suggested New Fields):")
    st.write(updated_target_schema)

    # Recalculate the similarity matrix with the updated target schema
    updated_target_embeddings = model.encode(updated_target_schema)
    updated_similarity_matrix = cosine_similarity(
        source_embeddings, updated_target_embeddings)
    updated_similarity_df = pd.DataFrame(
        updated_similarity_matrix, index=source_fields, columns=updated_target_schema)

    # Display updated similarity matrix
    st.write("Updated Similarity Matrix:")
    st.dataframe(updated_similarity_df)

    # Update field mapping with the new target schema
    updated_field_mapping = {}
    for source_field in updated_similarity_df.index:
        best_match = updated_similarity_df.loc[source_field].idxmax()
        similarity_score = updated_similarity_df.loc[source_field].max()
        updated_field_mapping[source_field] = (best_match, similarity_score)

    # Combine all source data into the updated target schema
    sources = [data[1] for data in source_data]
    unified_data = populate_target_schema(
        sources, updated_target_schema, updated_field_mapping, threshold)

    # Save the unified data to populated_target_schema.csv
    output_file = "./populated_target_schema.csv"
    unified_data.to_csv(output_file, index=False)
    st.write(f"Unified data saved to {output_file}")

    return updated_similarity_df, unified_data

# Function to populate the target schema with data from source schemas


def populate_target_schema(sources, target_schema, field_mapping, threshold):
    # Initialize a DataFrame with target schema columns
    unified_data = pd.DataFrame(columns=target_schema)

    # Populate data from source schemas
    for source_df in sources:
        for source_field in source_df.columns:
            if source_field in field_mapping:
                target_field, score = field_mapping[source_field]
                if score >= threshold:
                    # Map source data to target schema
                    unified_data[target_field] = source_df[source_field]

    # Fill missing columns with NaN
    unified_data = unified_data.reindex(columns=target_schema, fill_value=None)
    return unified_data


# Streamlit UI
st.title("Data migration poc")

# Upload source CSV files
uploaded_files = []
for i in range(3):
    uploaded_file = st.file_uploader(
        f"Upload source CSV file {i+1}", type="csv")
    if uploaded_file:
        uploaded_files.append(uploaded_file)

st.title("Target schema")
# Upload target schema CSV file
uploaded_target_file = st.file_uploader(
    "Upload target schema CSV file", type="csv")

# Process and display the similarity matrix and unified data when the button is pressed
if st.button("Process Data"):
    if uploaded_files and uploaded_target_file:
        similarity_matrix, unified_data = process_data(
            uploaded_files, uploaded_target_file)

        # Display the result
        if similarity_matrix is not None and unified_data is not None:
            st.write("Processed Data:")
            st.write("Similarity Matrix:", similarity_matrix)
            st.write("Unified Data:", unified_data)
    else:
        st.warning("Please upload the required CSV files.")
