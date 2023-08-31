"""
    In this file:
    1. preprocessing script which will create a searchable dataset from our corpus.
    2. function which will allow us to search through the corpus.
    
    ======
    
    I wonder if it would be a good idea to make this into a class.
    - You could initialize the class with the path to the dataset.
    - then you could have a "query" function which handles the querying.
    > seems like it would be a good interface...
"""

import pandas as pd
from pathlib import Path
import ast
from sentence_transformers import SentenceTransformer, util
import torch
import pickle as pkl

embedder = SentenceTransformer('all-MiniLM-L6-v2')


class corpus_sem_search:
    def __init__(self, corpus_data_path: Path, cache: bool=True):
        # option 1: send a pkl file with corpus info. Just unzip this.
        self.corpus_df = None
        self.corpus_embeddings = []
        if corpus_data_path.suffix == '.pkl':
            with open(corpus_data_path, 'rb') as pkl_in:
                self.corpus_df, self.corpus_embeddings = pkl.load(pkl_in)
                
        elif corpus_data_path.suffix == '.csv':
            self.corpus_df = self.create_corpus_df(corpus_data_path)
            self.corpus_embeddings = self.create_corpus_embeddings(
                                        list(self.corpus_df["sentence"]))
            # store output to pkl
            if cache:
                out_path = corpus_data_path.parent / Path(corpus_data_path.stem + '.pkl')
                with open(out_path, 'wb') as out_file:
                    pkl.dump((self.corpus_df, self.corpus_embeddings), out_file)
        
        else:
            raise Exception("Wrong filename suffix. Input file needs to be '.pkl'" 
                            "or '.csv'")
        

    def create_corpus_df(self, data_path: Path)->pd.DataFrame:
        # step 1: open up the pandas dataframe for the corpus. 
        #  convert format so that there is one row per sentence.
        
        # extract raw data
        df = pd.read_csv(data_path)
        
        # extract all the text from header, abstracts.
        def get_important_fields(row):
            # for any given row, get the most important fields (i.e. fields we are using).
            # Title text, Abstract text, publication #, application #
            title_field = ast.literal_eval(row["title_localized"])
            title_text = ""
            if title_field and len(title_field)>0:
              title_text = title_field[0]["text"]
        
            abstract_field = ast.literal_eval(row["abstract_localized"])
            abstract_text = ""
            if abstract_field and len(abstract_field)>0:
              abstract_text = abstract_field[0]["text"]
            
            all_text = title_text + abstract_text
            all_sentences = all_text.split('. ')
            
            output = []
            for s in all_sentences:
                output.append(
                        {"publication_num": row["publication_number"],
                          "application_num": row["application_number"],
                          "sentence": s}
                )
        
            return output
            
        # this return list of lists of corpus sentences
        out1 = df.apply(get_important_fields, axis=1)
        corpus_records = [item for sublist in out1 for item in sublist]
        # function output: one entry per corpus data point.
        return pd.DataFrame.from_records(corpus_records)


    # step 2: once you have the corpus, create corpus embeddings.
    def create_corpus_embeddings(self, corpus_sents: list):
        corpus_embeddings = embedder.encode(corpus_sents, convert_to_tensor=True)
        return corpus_embeddings
            
    
    # step 3: create a search function that returns the best matches.
    # note: global variables:
    #   1) corpus: list of corpus sentences
    #   2) corpus_embeddings: embeddings created from all the sentences.
    def search_corpus(self, query: str, k: int=10)->list:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        top_k = min(k, len(self.corpus_df))
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        # print results
        print("\n\n======================\n\n")
        print("Query:", query)
        print(f"\nTop {k} most similar documents in corpus:")
        
        # before doing this, refine so that only unique documents are returned.
        for score, idx in zip(top_results[0], top_results[1]):
            i = int(idx)
            patent_id = self.corpus_df.loc[i,"publication_num"]
            sentence = self.corpus_df.loc[i, "sentence"]
            print("(Patent id: {}, Score: {:.4f}, Sentence: {})".format(patent_id, score, sentence))
        
        return top_results


#---------------------------------------------------#
    
# try to test out the above. Create a semantic search
# object and get some results with a sample query.
if __name__ == "__main__":
    # first, load the search object
    cwd = Path("/Users/sidharth/Documents/ml/sideprojects/patent_semantic_search/")
    corpus_data_path = Path(cwd / "data/publications_1k.pkl")
    patent_search = corpus_sem_search(corpus_data_path, True)

    # next, try performing a query
    _ = patent_search.search_corpus("enhancement in electronic cigarettes or vaping")


    