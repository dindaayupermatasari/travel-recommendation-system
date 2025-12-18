from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__, static_folder='static')
CORS(app)

nltk.download('punkt')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

vectorizer = None
tfidf_matrix = None
data = None
load_error_message = None

try:
    vectorizer = pickle.load(open('model/tfidf_model.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('model/tfidf_matrix.pkl', 'rb'))
    data = pd.read_pickle('model/data_wisata.pkl')
except FileNotFoundError as e:
    load_error_message = (
        "Model atau data tidak ditemukan. "
        "Pastikan file model sudah tersedia di folder 'model'."
    )
    print(load_error_message)
except Exception as e:
    load_error_message = f"Terjadi kesalahan saat memuat model/data: {str(e)}"
    print(load_error_message)

def preprocess_text(text):
    """Preprocess the input text by lowercasing, removing punctuation, and stemming."""
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in list_stopwords]
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def get_recommendations(keyword, top_n=20, min_budget=None, max_budget=None, min_rating=None, kota=None, kategori=None):
    """Get recommendations based on the provided filters."""
    if data is None or vectorizer is None or tfidf_matrix is None:
        return {
            "success": False,
            "message": load_error_message or "Data/model belum siap."
        }

    preprocessed_keyword = preprocess_text(keyword)
    preprocessed_tokens = preprocessed_keyword.split()

    matches = data[data['nama_wisata'].str.lower().str.contains(preprocessed_keyword)]

    if not matches.empty:
        filtered_data = matches
    else:
        keyword_vector = vectorizer.transform([preprocessed_keyword])
        cosine_similarities = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
        similarity_scores = list(enumerate(cosine_similarities))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i for i, score in similarity_scores if score > 0][:top_n]
        filtered_data = data.iloc[top_indices]

    # Apply filters
    if min_budget is not None:
        filtered_data = filtered_data[filtered_data['harga_diskon'] >= min_budget]

    if max_budget is not None:
        filtered_data = filtered_data[filtered_data['harga_diskon'] <= max_budget]

    if min_rating is not None:
        filtered_data = filtered_data[filtered_data['rating'] >= min_rating]

    if kota is not None:
        filtered_data = filtered_data[filtered_data['kota'].str.lower() == kota.lower()]

    if kategori is not None:
        kategori_lower = kategori.lower()
        filtered_data = filtered_data[filtered_data['label'].str.lower().str.contains(kategori_lower)]

    if filtered_data.empty:
        return {
            "success": False,
            "message": "Tidak ada hasil yang sesuai dengan filter yang diterapkan"
        }

    filtered_data = filtered_data.sort_values(by=['rating', 'ulasan'], ascending=False)
    top_filtered = filtered_data.head(top_n)

    recommendations = []
    for idx in top_filtered.index:
        recommendations.append({
            "nama": top_filtered.loc[idx]["nama_wisata"],
            "harga_asli": float(top_filtered.loc[idx]["harga_asli"]),
            "harga": float(top_filtered.loc[idx]["harga_diskon"]),
            "rating": float(top_filtered.loc[idx]["rating"]),
            "ulasan": top_filtered.loc[idx]["ulasan"],
            "gambar": top_filtered.loc[idx]["gambar"],
            "deskripsi": top_filtered.loc[idx]["deskripsi"],
            "kota": top_filtered.loc[idx]["kota"],       
            "lokasi": top_filtered.loc[idx]["lokasi"] 
        })

    return {
        "success": True,
        "recommendations": recommendations
    }

@app.route('/')
def index():
    if data is None:
        return render_template('index.html', 
                               cities=[],
                               categories=[],
                               error_message=load_error_message)
    cities = data['kota'].unique().tolist()
    categories = data['label'].unique().tolist()
    categories = [cat for cat in categories if cat.lower() != "klinik kecantikan"]
    return render_template('index.html', cities=cities, categories=categories)

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    if data is None:
        return render_template('index.html', 
                               cities=[],
                               categories=[],
                               error_message=load_error_message)

    cities = data['kota'].unique().tolist()
    categories = data['label'].unique().tolist()
    categories = [cat for cat in categories if cat.lower() != "klinik kecantikan"]
    
    if request.method == 'POST':
        keyword = request.form.get('keyword', '')
        min_budget = request.form.get('min_budget', None)
        max_budget = request.form.get('max_budget', None)
        min_rating = request.form.get('min_rating', None)
        kota = request.form.get('kota', None)
        kategori = request.form.get('kategori', None)
        
        min_budget = float(min_budget) if min_budget else None
        max_budget = float(max_budget) if max_budget else None
        min_rating = float(min_rating) if min_rating else None
        kota = None if kota == "" else kota
        kategori = None if kategori == "" else kategori
        
        result = get_recommendations(
            keyword=keyword,
            top_n=20,
            min_budget=min_budget,
            max_budget=max_budget,
            min_rating=min_rating,
            kota=kota,
            kategori=kategori
        )
        
        return render_template('index.html', result=result, cities=cities, categories=categories)
    
    return render_template('index.html', cities=cities, categories=categories)

@app.route('/attractions', methods=['GET'])
def attractions():
    city = request.args.get('city')
    if not city:
        return jsonify({"success": False, "message": "City parameter is required"}), 400

    filtered_data = data[data['kota'].str.lower() == city.lower()]

    if filtered_data.empty:
        return jsonify({"success": False, "message": "No attractions found for the specified city"}), 404

    sorted_attractions = filtered_data.sort_values(by=['rating'], ascending=False)

    attractions_list = []
    for idx in sorted_attractions.index:
        attractions_list.append({
            "nama": sorted_attractions.loc[idx]["nama_wisata"],
            "harga": float(sorted_attractions.loc[idx]["harga_diskon"]),
            "rating": float(sorted_attractions.loc[idx]["rating"]),
            "ulasan": sorted_attractions.loc[idx]["ulasan"],
            "gambar": sorted_attractions.loc[idx]["gambar"],
            "deskripsi": sorted_attractions.loc[idx]["deskripsi"],
            "lokasi": sorted_attractions.loc[idx]["lokasi"]
        })

    return jsonify({"success": True, "attractions": attractions_list})

@app.route('/api/recommendations', methods=['GET'])
def api_recommend():
    if data is None:
        return jsonify({"success": False, "message": load_error_message or "Data/model belum siap."}), 500
    try:
        keyword = request.args.get('keyword', None)
        if not keyword:
            return jsonify({"success": False, "message": "Kata kunci diperlukan"}), 400
            
        top_n = request.args.get('top_n', 20, type=int)
        min_budget = request.args.get('min_budget', None, type=float)
        max_budget = request.args.get('max_budget', None, type=float)
        min_rating = request.args.get('min_rating', None, type=float)
        kota = request.args.get('kota', None)
        kategori = request.args.get('kategori', None)
        
        result = get_recommendations(
            keyword=keyword,
            top_n=top_n,
            min_budget=min_budget,
            max_budget=max_budget,
            min_rating=min_rating,
            kota=kota,
            kategori=kategori
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    
@app.route('/api/cities', methods=['GET'])
def api_cities():
    if data is None:
        return jsonify({"success": False, "message": load_error_message or "Data/model belum siap."}), 500

    city_list = data['kota'].unique().tolist()
    city_wisata = {}

    for city in city_list:
        city_data = data[data['kota'].str.lower() == city.lower()]
        sorted_city_data = city_data.sort_values(by='rating', ascending=False)

        wisata_list = []
        for _, row in sorted_city_data.iterrows():
            wisata_list.append({
                "nama": row["nama_wisata"],
                "harga": float(row["harga_diskon"]),
                "rating": float(row["rating"]),
                "ulasan": row["ulasan"],
                "gambar": row["gambar"],
                "deskripsi": row["deskripsi"],
                "lokasi": row["lokasi"]
            })

        city_wisata[city] = wisata_list

    return jsonify({"success": True, "data": city_wisata})

@app.route('/city/<city>')
def city_page(city):
    if data is None:
        return render_template('city.html', city=city, attractions=[], error_message=load_error_message)

    filtered_data = data[data['kota'].str.lower() == city.lower()]

    if filtered_data.empty:
        return render_template('city.html', city=city, attractions=[], error_message="No attractions found for this city.")

    sorted_attractions = filtered_data.sort_values(by=['rating'], ascending=False)

    attractions_list = []
    for idx in sorted_attractions.index:
        attractions_list.append({
            "nama": sorted_attractions.loc[idx]["nama_wisata"],
            "harga_asli": float(sorted_attractions.loc[idx]["harga_asli"]), 
            "harga": float(sorted_attractions.loc[idx]["harga_diskon"]),
            "rating": float(sorted_attractions.loc[idx]["rating"]),
            "ulasan": sorted_attractions.loc[idx]["ulasan"],
            "gambar": sorted_attractions.loc[idx]["gambar"],
            "deskripsi": sorted_attractions.loc[idx]["deskripsi"],
            "lokasi": sorted_attractions.loc[idx]["lokasi"]
        })

    return render_template('city.html', city=city, attractions=attractions_list)

if __name__ == '__main__':
    app.run(debug=True)
