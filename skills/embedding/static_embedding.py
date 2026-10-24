import gensim.downloader as api
import numpy as np
import logging

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
    
    # See the nearest words
    # print(kv.most_similar("cat", topn=5))


