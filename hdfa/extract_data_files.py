from data_wrangler import H5DataCreator

input_files = ["//Volumes//ExtShield//datasets//archive.zip",
			  "//Volumes//ExtShield//datasets//archive_2.zip"]

for input_file in input_files:
	file_name = input_file.split('/')[-1].split('.')[0]
	de = H5DataCreator.input_processor(input_file=input_file)

