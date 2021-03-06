import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import math


def get_acc(prediction,gt):
	count=0
	for v in range(len(gt)):
    	if prediction[v]==gt[v]:
        	count+=1
    return ((1.0*count)/len(gt))


def train():

	diabetes_data = pd.read_csv('pima-indians-diabetes-Copy1.csv')

	#diabetes_data.head()
	#diabetes_data.columns
	qantitative_cols = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps','Insulin', 'BMI', 'Pedigree','Age']

	#Normalize Quantitative Features - Can also use sklearn
	diabetes_data[quantitative_cols] = diabetes_data[quantitative_cols].apply(lambda x: (x-x.min()) / (x.max()-x.min()) )
	#diabetes_data.head()

	#Make numeric feature columns for tf estimator
	#numeric_column(key)
	'''
	key A unique string identifying the input feature. 
	It is used as the column name and the dictionary key
	for feature parsing configs, feature Tensor objects, 
	and feature columns)
	'''

	preg_feat = tf.feature_column.numeric_column(quantitative_cols[0])
	glu_feat = tf.feature_column.numeric_column(quantitative_cols[1])
	blood_feat = tf.feature_column.numeric_column(quantitative_cols[2])
	tri_feat = tf.feature_column.numeric_column(quantitative_cols[3])
	ins_feat = tf.feature_column.numeric_column(quantitative_cols[4])
	bmi_feat = tf.feature_column.numeric_column(quantitative_cols[5])
	ped_feat = tf.feature_column.numeric_column(quantitative_cols[6])

	#TODO - Try making age a categorical feature column without any normalization 
	age_feat = tf.feature_column.numeric_column(quantitative_cols[7])


	#Make Categorical feature columns for tf estimator
	#hash bucket  = automatic categories
	group_feat = tf.feature_column.categorical_column_with_hash_bucket('Group' , hash_bucket_size=4)

	#Age Feat as categorical - do without normalization to age
	age_feat_cat = tf.feature_column.bucketized_column(tf.feature_column.numeric_column('Age'),boundaries=[20,30,40,50,60,70,80])


	feat_list = [preg_feat,glu_feat,blood_feat,tri_feat,ins_feat,bmi_feat,ped_feat,,group_feat,age_feat]


	#Make Train-Test Split
	x_data = diabetes_data.drop('Class',1)
	y_label = diabetes_data['Class']

	from sklearn.model_selection import train_test_split
	x_train_temp, x_test,y_train_temp, y_test = train_test_split(x_data,y_label,test_size=0.1)
	x_train, x_val,y_train, y_val = train_test_split(x_train_temp,y_train_temp,test_size=0.2)


	# Using DNN
	#Categorical Column need to be made into embedding column
	#dims = no of groups in column
	embedded_group_feat = tf.feature_column.embedding_column(group_feat,dimension=4)
	feat_list_dense = [preg_feat,glu_feat,blood_feat,tri_feat,ins_feat,bmi_feat,ped_feat,embedded_group_feat,age_feat]

	dense_model = tf.estimator.DNNClassifier(hidden_units=[100,200,50],feature_columns=feat_list_dense,n_classes=2)

	dense_train_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=10,num_epochs=200,shuffle=True)

	dense_model.train(input_fn=dense_train_func,steps=1000)

	dnse_eval_func = tf.estimator.inputs.pandas_input_fn(x=x_val,y=y_val,batch_size=10,num_epochs=1,shuffle=False)

	dense_val_result = dense_model.evaluate(input_fn=dense_eval_func)

	dense_test_func = tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=10,num_epochs=1,shuffle=False)

	dense_model_preds = list(model.predict(input_fn=dense_test_func))

	final_dense_preds = [int (i['class_ids']) for i in dense_model_preds]

	dense_test_acc = get_acc(final_dense_preds,y_test)

	print ("Test acc is %.2f" %(dense_test_acc))

if __name__ == '__main__':

	train()
