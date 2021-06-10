jupyter:
	jupyter notebook --no-browser --port 8888 --allow-root --NotebookApp.token='' --NotebookApp.password=''
play:
	@echo "🐋 Started"
	@@time(\
		time python playground.py \
	)
	@echo -e "\n🐬 Finished\n"
