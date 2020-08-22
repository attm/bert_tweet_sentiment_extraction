from setuptools import setup, find_packages


setup(
   name='bert_semantic_analysis',
   version='1.0',
   description='Personal project package',
   author='atgm1113',
   author_email='atgm1113@gmail.com',
   packages=find_packages(include=["src", 
                                   "src.data_process",
                                   "src.text_process", 
                                   "src.main", 
                                   "src.model", 
                                   "src.tests", 
                                   "src.utils", 
                                   "src.",
                                   "src.bert_semantic_classifier"], 
                                   exclude=[]))