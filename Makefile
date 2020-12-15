jupyter:
	jupyter notebook --no-browser --port 8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''
cuda:
	@@time(\
		time python entrypoint.py \
	)
	@echo -e "\n🐬 Finished\n"
cpu:
	@@time(\
		time python potential_python.py \
	)
	@echo -e "\n🐬 Finished\n"
