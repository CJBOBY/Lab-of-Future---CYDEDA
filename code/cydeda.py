import pandas as pd
import ast
from sklearn.cluster import KMeans
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import google.generativeai as genai

import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller .exe """
    try:
        base_path = sys._MEIPASS  # When running in .exe
    except Exception:
        base_path = os.path.abspath(".")  # When running from .py

    return os.path.join(base_path, relative_path)


# --- Gemini Setup ---
genai.configure(api_key="AIzaSyAuTSYcMEt5Xg1lTUGzmkYPx2ow026Wv5s")
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

def chat_with_gemini(user_input):
    try:
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- Load Data ---
movies_url = 'https://cbscabushooudsevqytf.supabase.co/storage/v1/object/public/datas//tmdb_5000_movies%20(1).csv'
credits_url = 'https://cbscabushooudsevqytf.supabase.co/storage/v1/object/public/datas//tmdb_5000_credits.csv'

credits = pd.read_csv(credits_url, on_bad_lines='skip')
movies = pd.read_csv(movies_url, on_bad_lines='skip')
movies = movies.merge(credits, on='title')
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

# --- Feature Extraction Functions ---
def parse_features(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)] if isinstance(x, str) else []
    except:
        return []

def extract_top_3_cast(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)][:3] if isinstance(x, str) else []
    except:
        return []

def extract_director_and_composer(x):
    director, composer = None, None
    try:
        for member in ast.literal_eval(x):
            if member.get('job') == 'Director':
                director = member['name']
            if member.get('job') == 'Original Music Composer':
                composer = member['name']
    except:
        pass
    return director, composer

movies['genres'] = movies['genres'].apply(parse_features)
movies['keywords'] = movies['keywords'].apply(parse_features)
movies['cast'] = movies['cast'].apply(extract_top_3_cast)
movies['director'], movies['composer'] = zip(*movies['crew'].apply(extract_director_and_composer))
all_genres = sorted({genre for sublist in movies['genres'] for genre in sublist})

def encode_genres(movies):
    return [[1 if genre in genres else 0 for genre in all_genres] for genres in movies['genres']]

def get_movie_features(movies):
    genre_matrix = encode_genres(movies)
    features = pd.DataFrame(genre_matrix, columns=all_genres)
    features['runtime'] = movies['runtime'].fillna(0)
    features['vote_average'] = movies['vote_average'].fillna(0)
    features['release_year'] = movies['release_date'].dt.year.fillna(0).astype(int)
    return features

def apply_clustering(features):
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(features)
    features['cluster'] = kmeans.predict(features)
    distances = kmeans.transform(features.drop(columns=['cluster']))
    closest_distances = np.min(distances, axis=1)
    max_dist = np.max(closest_distances)
    features['confidence'] = 100 * (1 - closest_distances / max_dist)
    features['confidence'] = np.clip(features['confidence'], 0, 100)
    return features, kmeans

movie_features = get_movie_features(movies)
movie_features, kmeans = apply_clustering(movie_features)

# --- GUI Application ---
class MovieRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommender")
        self.root.configure(bg="#2e3440")
        self.logo_img = None
        self.page1()


    def page1(self):
        self.clear()
        try:
            if not self.logo_img:
                img = Image.open(resource_path("logo.png")).resize((500, 400))
                self.logo_img = ImageTk.PhotoImage(img)
            label = tk.Label(self.root, image=self.logo_img, bg="#2e3440")
            label.pack(pady=10)
        except Exception:
            pass
        next_btn = tk.Button(self.root, text="Next", command=self.page2, font=("Helvetica", 12, "bold"),
                            bg="#5e81ac", fg="white", activebackground="#81a1c1", activeforeground="white")
        next_btn.pack(pady=10)


    def page2(self):
        self.clear()
        # Left frame for inputs
        left_frame = tk.Frame(self.root, bg="#3b4252", padx=15, pady=15)
        left_frame.pack(side="left", fill="y")

        # Right frame for output text box
        right_frame = tk.Frame(self.root, bg="#4c566a", padx=15, pady=15)
        right_frame.pack(side="right", fill="both", expand=True)

        # Genres Label
        tk.Label(left_frame, text="Select Genres:", bg="#3b4252", fg="white", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky='w', pady=(0,5))

        genre_frame = tk.Frame(left_frame, bg="#3b4252")
        genre_frame.grid(row=1, column=0, sticky='w')

        self.genre_vars = {}
        for i, genre in enumerate(all_genres):
            var = tk.BooleanVar()
            chk = tk.Checkbutton(genre_frame, text=genre, variable=var, bg="#3b4252", fg="white", selectcolor="#81a1c1", activebackground="#81a1c1")
            chk.grid(row=i // 4, column=i % 4, sticky='w', padx=5, pady=2)
            self.genre_vars[genre] = var

        # Entry helper
        def make_entry(label, row):
            tk.Label(left_frame, text=label, bg="#3b4252", fg="white").grid(row=row, column=0, sticky='w', pady=3)
            entry = tk.Entry(left_frame)
            entry.grid(row=row, column=1, pady=3)
            return entry

        self.runtime_entry = make_entry("Max Runtime (minutes):", 2)
        self.vote_entry = make_entry("Min Vote Average:", 3)
        self.start_year_entry = make_entry("Start Year:", 4)
        self.end_year_entry = make_entry("End Year:", 5)
        self.prod_entry = make_entry("Production Company:", 6)
        self.cast_entry = make_entry("Cast (comma-separated):", 7)
        self.dir_entry = make_entry("Director:", 8)
        self.comp_entry = make_entry("Composer:", 9)
        self.similar_title_entry = make_entry("Movies (comma-separated):", 10)  # New input field for multiple movies

        # Buttons
        submit_btn = tk.Button(left_frame, text="Recommend Movies (Filters)", command=self.on_submit,
                               bg="#81a1c1", fg="white", activebackground="#88b0f7", font=("Helvetica", 12, "bold"))
        submit_btn.grid(row=11, column=0, columnspan=2, pady=10, sticky='we')

        multi_recommend_btn = tk.Button(left_frame, text="Recommend by Movies ",
                                        command=self.recommend_by_multiple_movies_and_cast,
                                        bg="#5e81ac", fg="white", activebackground="#81a1c1", font=("Helvetica", 12, "bold"))
        multi_recommend_btn.grid(row=12, column=0, columnspan=2, pady=5, sticky='we')

        # Output Text
        self.output_text = tk.Text(right_frame, height=30, width=80, bg="#d8dee9", fg="#2e3440", font=("Helvetica", 11))
        self.output_text.pack(fill="both", expand=True)

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def on_submit(self):
        try:
            selected_genres = [g for g, var in self.genre_vars.items() if var.get()]
            runtime = int(self.runtime_entry.get()) if self.runtime_entry.get() else None
            vote_avg = float(self.vote_entry.get()) if self.vote_entry.get() else None
            start_yr = int(self.start_year_entry.get()) if self.start_year_entry.get() else None
            end_yr = int(self.end_year_entry.get()) if self.end_year_entry.get() else None
            prod_company = self.prod_entry.get()
            cast_input = self.cast_entry.get()
            director = self.dir_entry.get()
            composer = self.comp_entry.get()

            filters = pd.Series(True, index=movies.index)
            if selected_genres:
                filters &= movies['genres'].apply(lambda x: any(g.lower() in [g2.lower() for g2 in x] for g in selected_genres))
            if runtime:
                filters &= movies['runtime'].notna() & (movies['runtime'] <= runtime)
            if vote_avg:
                filters &= movies['vote_average'].notna() & (movies['vote_average'] >= vote_avg)
            if start_yr and end_yr:
                filters &= movies['release_date'].apply(lambda x: start_yr <= x.year <= end_yr if pd.notna(x) else False)
            if prod_company:
                filters &= movies['production_companies'].apply(lambda x: prod_company.lower() in str(x).lower())
            if cast_input:
                cast_list = [c.strip().lower() for c in cast_input.split(',')]
                filters &= movies['cast'].apply(lambda x: any(c in [a.lower() for a in x] for c in cast_list))
            if director:
                filters &= movies['director'].apply(lambda x: director.lower() in str(x).lower() if pd.notna(x) else False)
            if composer:
                filters &= movies['composer'].apply(lambda x: composer.lower() in str(x).lower() if pd.notna(x) else False)

            result = movies[filters]
            if result.empty:
                result_text = "No matching movies found."
            else:
                result = result.copy()
                result['confidence'] = movie_features.loc[result.index, 'confidence']
                result = result.sort_values(by='confidence', ascending=False).head(10)
                result_text = "🎬 Recommended Movies:\n\n"
                for _, row in result.iterrows():
                    year = row['release_date'].year if pd.notna(row['release_date']) else 'Unknown'
                    result_text += (
                        f"🎬 {row['title']} ({year})\n"
                        f"⭐ Rating: {row['vote_average']}\n"
                        f"🕒 Runtime: {int(row['runtime']) if pd.notna(row['runtime']) else 'N/A'} mins\n"
                        f"🎯 Confidence: {row['confidence']:.2f}%\n"
                        "-------------\n"
                    )

                titles = [row['title'] for _, row in result.iterrows()]
                gemini_prompt = "Give a brief overview of the following movies:\n" + ", ".join(titles)
                overview = chat_with_gemini(gemini_prompt)
                result_text += "\n🎥 Movie Overviews:\n" + overview

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def recommend_by_multiple_movies_and_cast(self):
        try:
            movie_titles_input = self.similar_title_entry.get().strip()
            cast_input = self.cast_entry.get().strip()

            if not movie_titles_input and not cast_input:
                messagebox.showinfo("Input Needed", "Please enter at least one movie title or cast name.")
                return

            movie_titles = [t.strip().lower() for t in movie_titles_input.split(",") if t.strip()]
            cast_names = [c.strip().lower() for c in cast_input.split(",") if c.strip()]

            cast_from_movies = set()
            for title in movie_titles:
                match = movies[movies['title'].str.lower() == title]
                if not match.empty:
                    cast_list = match.iloc[0]['cast']
                    cast_from_movies.update([c.lower() for c in cast_list if isinstance(c, str)])

            combined_cast = set(cast_names) | cast_from_movies

            if not combined_cast:
                messagebox.showinfo("No Cast Found", "No cast found from the provided movies or cast input.")
                return

            def has_common_cast(cast_list):
                return any(c.lower() in combined_cast for c in cast_list if isinstance(c, str))

            filtered = movies[movies['cast'].apply(has_common_cast)]

            selected_genres = [g for g, var in self.genre_vars.items() if var.get()]
            if selected_genres:
                filtered = filtered[filtered['genres'].apply(lambda x: any(g.lower() in [gg.lower() for gg in x] for g in selected_genres))]

            if filtered.empty:
                result_text = "No matching movies found based on cast and genre."
            else:
                filtered = filtered.copy()
                filtered['confidence'] = movie_features.loc[filtered.index, 'confidence']
                filtered = filtered.sort_values(by='confidence', ascending=False).head(10)

                result_text = "🎬 Recommended Movies Based on Cast Priority:\n\n"
                for _, row in filtered.iterrows():
                    year = row['release_date'].year if pd.notna(row['release_date']) else 'Unknown'
                    result_text += (
                        f"🎬 {row['title']} ({year})\n"
                        f"⭐ Rating: {row['vote_average']}\n"
                        f"🕒 Runtime: {int(row['runtime']) if pd.notna(row['runtime']) else 'N/A'} mins\n"
                        f"🎯 Confidence: {row['confidence']:.2f}%\n"
                        "-------------\n"
                    )

                titles = [row['title'] for _, row in filtered.iterrows()]
                gemini_prompt = "Give a brief overview of the following movies:\n" + ", ".join(titles)
                overview = chat_with_gemini(gemini_prompt)
                result_text += "\n🎥 Movie Overviews:\n" + overview

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

# --- Run App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommenderApp(root)
    root.mainloop()
