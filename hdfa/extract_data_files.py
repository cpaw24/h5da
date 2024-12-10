from data_wrangler import H5DataExtractor

input_files = ["//Volumes//ExtShield//datasets//archive.zip",
			  "//Volumes//ExtShield//datasets//archive_2.zip"]

for input_file in input_files:
	file_name = input_file.split('/')[-1].split('.')[0]
	de = H5DataExtractor(output_file=f"data_{file_name}.h5", input_file=input_file, input_dict={})

