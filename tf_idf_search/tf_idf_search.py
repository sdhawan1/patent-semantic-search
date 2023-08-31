"""
  In this file: 
    - start out with a tf-idf benchmark, coded in python.
    - algorithm:
      - performs simple tf-idf of the query with every document in the corpus.
      - Then, it sorts the documents by the tf-idf score and returns the top
        N documents.
"""

import spacy
import math
import pandas as pd
from pathlib import Path
import ast
import pickle as pkl


#--------------------------------------------#
# helper functions & objects:
#--------------------------------------------#

nlp = spacy.load("en_core_web_lg")


def tokenize_filter_stopwords(text: str)->list:
    # given a piece of text, perform preprocessing to 
    # filter out non-word tokens and filter out common
    # stopwords.
    nt = nlp(text)
    query_words = []
    for token in nt:
        tl = token.lemma_
        if token.is_alpha and not token.is_stop:
            query_words.append(tl)
    return query_words


def preprocess_data(df_1k:pd.DataFrame)->pd.DataFrame:
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

        fields = {"title": title_text, "abstract": abstract_text, 
                  "publication_num": row["publication_number"],
                  "application_num": row["application_number"]}

        return fields
        

    psd = df_1k.apply(get_important_fields, axis=1)
    patent_search_data = pd.DataFrame.from_records(psd)

    # create a field with all combined text...
    def combine_text(row):
        return row["title"] + ".\n" + row["abstract"]
    patent_search_data["all_text"] = patent_search_data.apply(combine_text, axis=1)

    #patent_search_data.head()
    return patent_search_data

#--------------------------------------------#
# first, extract all tf's (term frequencies):
#--------------------------------------------#


def bag_of_words(text: str)->dict:   
    # for every document, find counts of all important lemmas.

    # step 1: get all the distinct lemmas:
    bow = {}
    query_words = tokenize_filter_stopwords(text)

    # create a dictionary of word counts.
    for w in query_words:
        w = str(w)
        if w in bow:
            bow[w] += 1
        else:
            bow[w] = 1
    
    return bow


def create_document_tfs(
    patent_search_data: pd.DataFrame)->dict:
    """
       Given our input patent semantic search dataset, create a dictionary
       mapping every patent id to its term frequencies.        
    """

    document_tf_dict = {}
    for rowind in range(patent_search_data.shape[0]):
        row = patent_search_data.loc[rowind, :]
        key = row["publication_num"]

        # find the bow
        bow = bag_of_words(row["all_text"])
        document_tf_dict[key] = bow
    
    return document_tf_dict


def create_document_idfs(document_tf_dict: dict)->dict:
    # within our dataset, establish document "idf's" for every lemma.

    lemma_df = {}
    # compute "document frequency" for all lemmas in document set.
    for bow in document_tf_dict.values():
        for lemma in bow.keys():
            if lemma in lemma_df:
                lemma_df[lemma] += 1
            else:
                lemma_df[lemma] = 1
    
    # compute all the lemma idf's: log(N/df)
    N = len(document_tf_dict)  # this field is the total # of docs.
    lemma_idf = {}
    for key in lemma_df.keys():
        df = lemma_df[key]
        lemma_idf[key] = math.log10(N/(df+1))
    
    return lemma_idf

#--------------------------------------------#
# Next, perform the search
#--------------------------------------------#

"""
  Next steps: 
  1. write a script to calculcate tf-idf for a particular query +
  one of the columns in the dataframe.

  2. pre-calculate all word counts in the table in order to speed this up.
"""


def tf_idf_demo(query:str, publ_no:str,
           doc_tf_dct: dict, lemma_idfs: dict)->list:
    """
      First, a script to do tf-idf for a single query and document.
      Note: this function won't perform a full search, it will just
        give a demo score to any document you want.
      
      query: search string
      publ_no: publication number (effectively the document's id)
      doc_tf_dct: data structure (created in "create_document_tfs") 
        which maps: {docid -> {lemma -> term frequency}}
      lemma_idfs: data structure (created in "create_document_idfs")
        which maps: {lemma -> idf score for the corpus}

      example usage:
      >> import pandas as pd
      >> df = pd.read_csv(publications_1k.csv)
      ~~ format the data...
      >> doc_tf_dct = create_document_tfs(df)
      >> lemma_idfs = create_document_idfs(doc_tf_dct)
      >> print(tf_idf("cigarette", "US-9462831-B2", doc_tf_dct, lemma_idfs))
    """
    
    # step 1: identify the distinct & important words in the query.
    nq = nlp(query)
    query_words = []
    for token in nq:
        tl = token.lemma_
        if token.is_alpha and not token.is_stop:
            query_words.append(tl)
    query_words = list(set(query_words))
    print(query_words)

    # step 2: get all term frequencies and inverse-document frequencies
    # desired data structure: {term: (tf, idf)}
    # assume another data structure "documents" containing "docid -> nlp(document), word_counts: {lemma: frequency}"
    document_tfs = doc_tf_dct[publ_no]
    tfidf_values = {}
    for lemma in query_words:
        tf = document_tfs.get(lemma, 0)
        idf = lemma_idfs[lemma]
        tfidf_values[lemma] = (tf, idf)

    # step 3: combine them all together.
    tf_idf = sum([tf * idf for tf, idf in tfidf_values.values()])
    return tf_idf


def tf_idf_search(query:str, dataset_path:Path, 
                  N:int=10, cache:bool=True)->pd.DataFrame:
    """
      Perform the full query in this function.
      - given a dataset file and a query, return the objects in 
        the dataset that most closely match the query.
      
      - v0: pass the query and the path to the dataset.
    """
    
    # save the data, tf's and idf's as global variables:
    print("Preprocessing data and indexing dataset...")
    cache_path = dataset_path.parents[0] / dataset_path.stem
    patent_search_data = None; doc_tf_dct = None; lemma_idfs = None
    if not cache_path.exists():
        df = pd.read_csv(dataset_path)
        patent_search_data = preprocess_data(df)
        doc_tf_dct = create_document_tfs(patent_search_data)
        lemma_idfs = create_document_idfs(doc_tf_dct)
        print("done")
        if cache:
            print("Caching tf-idf 'index' for later use...")
            cache_path.mkdir()
            patent_search_data.to_csv(cache_path / "patent_search_data.csv")
            with open(cache_path / "doc_tf_dct.pkl" ,"wb") as fout:
                pkl.dump(doc_tf_dct, fout)
            with open(cache_path / "lemma_idfs.pkl" ,"wb") as fout:
                pkl.dump(lemma_idfs, fout)
            print("done")
          
    # adding caching to save time.
    else:
        print("Retrieving index from cache...")
        patent_search_data = pd.read_csv(cache_path / "patent_search_data.csv")
        with open(cache_path / "doc_tf_dct.pkl", "rb") as fin:
            doc_tf_dct = pkl.load(fin)
        with open(cache_path / "lemma_idfs.pkl", "rb") as fin:
            lemma_idfs = pkl.load(fin)
        print("done")

    # break the query down into important lemmas
    nq = nlp(query)
    query_words = []
    for token in nq:
        tl = token.lemma_
        if token.is_alpha and not token.is_stop:
            query_words.append(tl)
    query_words = list(set(query_words))
    print("important query words:", query_words)

    # retrieve all tf_idf scores
    print("Searching dataset to find best matches...")
    tf_idf_scores = []
    for publ_no in list(patent_search_data["publication_num"]):
        document_tfs = doc_tf_dct[publ_no]
        tfidf_score = 0
        for lemma in query_words:
            tf = document_tfs.get(lemma, 0)
            idf = lemma_idfs.get(lemma, 0)
            tfidf_score += tf * idf
        tf_idf_scores.append(tfidf_score)
    
    # sort dataframe by scores
    #breakpoint()
    patent_search_data["search_score"] = tf_idf_scores
    psd = patent_search_data.sort_values("search_score", ascending=False).reset_index()
    psd_n = psd.head(N)
    print("\n#-------------------------------------------#\n")
    for r in range(N):
        row = psd.loc[r,:]
        print(f"Result #{r}:\n\tTitle: {row['title']}\n\tPublication#: {row['publication_num']}")
    print("\n")
    return psd_n


if __name__ == "__main__":
    dataset_path = Path("../data/publications_1k.csv")
    cwd = Path('.')
    p = cwd / dataset_path
    query = "semiautomatic rifle gun"
    
    # generate results
    output_df = tf_idf_search(query, p)
    
    # reformat output df...
    output_df_2 = pd.DataFrame({
    	"patent_id": output_df["publication_num"],
    	"score": output_df["search_score"],
    	"title": output_df["title"]
    })
    
    # store results
    output_file_name = query.replace(" ", "_")
    output_path = cwd / f"../results/{output_file_name}.csv"
    output_df_2.to_csv(output_path)



