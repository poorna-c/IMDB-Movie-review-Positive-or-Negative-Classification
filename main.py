from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
import os

word_indexes = imdb.get_word_index()
word_indexes = {k:(v+2) for k,v in word_indexes.items()}
word_indexes["<PAD>"] = 0
word_indexes["<START>"] = 1
word_indexes["<UKN>"] = 2
reverse_word_indexes = {v:k for k,v in word_indexes.items()}

def load_data():
	(X1,y1),(X2,y2) = imdb.load_data()
	return X1,y1,X2,y2


def preprocess_data(train_data,test_data):
	train_data = keras.preprocessing.sequence.pad_sequences(train_data,maxlen=512,value = 0, padding='post')
	test_data = keras.preprocessing.sequence.pad_sequences(test_data,maxlen=512,value = 0, padding = 'post')
	return train_data,test_data

def load_model():
	model = Sequential()
	model.add(Embedding(80000,16))
	model.add(GlobalAveragePooling1D())
	model.add(Dense(16, activation = 'relu'))
	model.add(Dense(1, activation = 'sigmoid'))
	model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['acc'])
	return model

def train():
	print("\nLoading your datasets...")
	train_data,train_labels,test_data,test_labels = load_data()
	print("\nLoading datasets success...")
	print("\nPreparing your data...")
	train_data, test_data = preprocess_data(train_data,test_data)
	print("\nPreprocessing on data success...")
	print("\nLoading your model...")
	model = load_model()
	print("\nLoading model success...")
	print("\npress any key to start training your model...")
	os.system('pause')
	os.system('cls')
	model.fit(train_data,train_labels,epochs = 15, batch_size = 32, validation_split = 0.2)
	os.system('cls')
	print('press any key to continue...(Your trained model will be trained automatically...)')
	os.system('pause')
	print("\nSaving your model...")
	model.save('model.h5')
	print("\nModel Saved successfully...")
	print("\nModel Evaluation in test set :")
	model.evaluate(test_data,test_labels)
	print("\nModel Details : ")
	model.summary()
	os.system('pause')
	print("\nExiting...")
	exit()

def decode_review(text):
	return " ".join([reverse_word_indexes.get(i, "?") for i in text])

def encode_review(text):
	encoded = [1]
	for word in text:
		if word.lower() in word_indexes:
			encoded.append(word_indexes[word.lower()])
		else:
			encoded.append(2)
	return encoded

if __name__ == '__main__':
	if(os.path.isfile('model.h5')):
		choice = input("We have found your pretrained model, Do you want to retrain your model...?(Y/N)(Recomended:N) :")
		if choice.upper() == "Y":
			train()
		else:
			os.system('cls')
			print("\nLoading model...")
			model = keras.models.load_model('model.h5')
			os.system('cls')
			print("\nModel loaded successfully...\n\n")
			review = input("Enter User Review : ")
			review = review.replace(".","").replace(",","").replace("?","").replace("\"","").replace("(","").replace(")","").replace("-","").strip().split(" ")
			review = encode_review(review)
			review = keras.preprocessing.sequence.pad_sequences([review],maxlen=512,value = 0, padding='post')
			print(review,decode_review(review[0]))
			prediction = model.predict(review[0].reshape(1,-1))
			print("\nPrediction : ",prediction[0])

	else:
		train()