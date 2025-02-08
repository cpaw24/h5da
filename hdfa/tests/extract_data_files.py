from hdfa.dataWrangler import DataProcessor

# Extract constants for hardcoded paths to make the code more maintainable
INPUT_FILE_PATH = "//Volumes//ExtShield//datasets//fashion-dataset.zip"
SCHEMA_FILE_PATH = "//Users//claude_paugh//PycharmProjects//HDF5//hdfa//schema//file_schema.json"
OUTPUT_DIR = "//Volumes//ExtShield//datasets//hdfa_files//"
BATCH_PROCESS_LIMIT = 5

def initialize_data_processor(input_file: str, schema_file: str, output_dir: str) -> DataProcessor:
	"""
    Initializes the DataProcessor with the given parameters.
    """
	file_name = input_file.split("/")[-1].split(".")[0]  # Extract file name without extension
	output_file = f"{output_dir}/{file_name}_filestore.h5"
	return DataProcessor(
		input_file=input_file,
		input_dict={},
		output_file=output_file,
		schema_file=schema_file
	)


# Main logic
data_processor = initialize_data_processor(INPUT_FILE_PATH, SCHEMA_FILE_PATH, OUTPUT_DIR)
data_processor.file_processor(group_keys=["images/", "styles/"], batch_process_limit=BATCH_PROCESS_LIMIT)


