import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

def new_line(new_review):
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = loaded_vectorizer.transform(new_corpus).toarray()
    predictions = int(model.predict(new_X_test))
    return predictions

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #this_string = [str(x) for x in request.form.values()]
    #this_string = str(request.args.get("tweet_text"))
    #this_string = [str(x) for x in request.form.values()]
    #print(this_string)
    #final_features = [np.array(int_features)]
    this_string = request.form['tweet_text']
    predictions = new_line(this_string)
    if predictions == 1:
        feelings="positive"
    if predictions == 0:
        feelings="negative"
    

    return render_template('index.html', prediction_text='The sentiment of the above twitter user is '+feelings)

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)