import gensim.downloader as api
import numpy as np
import logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configure logging to write to a file and track INFO level or higher
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# This model only use Lowercase words

def load_model(model_name : str):
    try :
        logging.info("Importing the model")
        # import the kv 50 dimention model
        model = api.load(model_name)
        print(model)
        return model
    except Exception as e:
        print(f"Something went wrong {e}")


def get_info():
    logging.info("Geting infos about the available models in gensim")
    # get the info api
    info = api.info()
    # extract the available models into a list
    models_list = list(info["models"].keys())
    for m in models_list:
        print(m)


def cosine_simularity(vector_a : np.ndarray, vector_b : np.ndarray):
    norm_a = np.linalg.norm(vector_a) 
    norm_b = np.linalg.norm(vector_b)
    return np.dot(vector_a, vector_b) / (norm_a * norm_b)

def dot_product(vector_a : np.ndarray, vector_b : np.ndarray):
    return np.dot(vector_a, vector_b)

def get_details(vector , word: str)-> None:
    logging.info(f"Details about\n {word} ")
    # Get the embedding of the word 
    print(vector)
    # Get the type of the embedding vector
    print(type(vector))
    # Dimension is 1
    print(vector.ndim)
    # The size is 50 and depend on the embedding model
    print(vector.size)
    # The type of each element in the embedding vector
    print(vector.dtype)

def plot_vectors(vector_a, vector_b, label_a="Word A", label_b="Word B"):

    # We project onto the 2D plane that contains
    # both vectors. Two vectors always lie in one plane, so the lengths and the
    # angle between them stay exactly the same as in the original 50 dimensions.

    # First axis: the direction of vector_a (divided by its length -> unit vector)
    e1 = vector_a / np.linalg.norm(vector_a)

    # Second axis: the part of vector_b that is perpendicular to e1.
    # We remove from vector_b its component along e1, what is left is at 90 degrees
    b_perp = vector_b - np.dot(vector_b, e1) * e1
    e2 = b_perp / np.linalg.norm(b_perp)

    # 2D coordinates of each vector = its dot product with each axis.
    # vector_a ends up at (|a|, 0), vector_b at (its projection on e1, on e2)
    vectors_2d = np.array([
        [np.dot(vector_a, e1), np.dot(vector_a, e2)],
        [np.dot(vector_b, e1), np.dot(vector_b, e2)],
    ])

    plt.figure(figsize=(6, 6))
    
    # Define origins for both arrows starting at (0,0)
    X = [0, 0]
    Y = [0, 0]

    # Vector directions (U, V) from the 2D reduced arrays
    U = vectors_2d[:, 0]
    V = vectors_2d[:, 1]

    # Use quiver to draw them as arrows originating from (0,0)
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=['coral', 'dodgerblue'])

    # Map the text labels dynamically
    words = [label_a, label_b]
    for i, word in enumerate(words):
        # Offset the text slightly from the arrow tip for readability
        plt.text(U[i] * 1.05, V[i] * 1.05, word, fontsize=12, weight='bold', ha='center')

    # Adjust plot boundaries to fit the arrows beautifully
    max_val = np.max(np.abs(vectors_2d)) * 1.3
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)

    # Add a grid, axis lines through the origin, and titles
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title(f"2D Projection of '{label_a}' and '{label_b}' GloVe Vectors", fontsize=14, pad=15)
    # The axes are the plane axes built above, not principal components
    plt.xlabel(f"Axis 1 (direction of '{label_a}')")
    plt.ylabel("Axis 2 (perpendicular to axis 1)")

    # Render the plot
    plt.show()



if __name__ == "__main__":
    # get_info()
    kv = load_model("glove-wiki-gigaword-50")
    get_details(kv["king"], "king")
    get_details(kv["dog"], "dog")

    # Unit length vector (normalised): the original vector divided by its own length
    # (L2 norm), so its length becomes exactly 1 while its direction stays the same.
    # Only the direction (the meaning) is kept, the magnitude is dropped. With unit
    # vectors, the dot product equals the cosine similarity, which makes comparing
    # words simpler and faster.

    kv_norm = kv.get_vector("king", norm=True)
    get_details(kv_norm, "king")

    

    # verify if a word is known 
    print(f"Is the word king known in the embedding model ? {"king" in kv}")
    print(f"Is the word king known in the embedding model ? {"Bambouzeling" in kv}")

    # Number of words
    len(f"This is the length of kv {len(kv)}")

    # display all words (change to true to see)
    if False:
        print(kv.index_to_key)

    # calculate the cosign simularity of two words
    print(f"kv cosign simularity {kv.similarity("cat", "dog")}")

    print(f"Np cosign simularity {cosine_simularity(kv["cat"] , kv["dog"])}")

    print(f"Np dot producr {dot_product(kv["cat"] , kv["dog"])}")


    if False :
        plot_vectors(kv["dog"], kv["cat"], "dog" , "cat")
        plot_vectors(kv["king"], kv["queen"], "king" , "queen")
        plot_vectors(kv["king"], kv["man"], "king" , "man")

    # See the nearest words
    print(kv.most_similar("cat", topn=5))

    # Word arithmetic: king - man + woman
    result = kv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
    print(result)



