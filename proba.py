from Classifiers import Classifiers

models = Classifiers()
print(models.listClassifiers())
if 'KNN1' not in models.models.keys():
	print('netacno')
else:
	print('tacno')