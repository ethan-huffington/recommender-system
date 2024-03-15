
# 🔵🔵🔵🔵
import subprocess
import sys

# Use subprocess to run the pip install command
# subprocess.check_call([sys.executable, "-m", "pip", "install", "pinecone-client"])

import pinecone
from pinecone import Pinecone

import os
import pandas as pd

import hopsworks
from opensearchpy import OpenSearch

import logging

class Transformer(object):
    
    def __init__(self):
        """
        Make Connections and Assemble Input Components
        - articles view
        - customers view
        - candidate (item) embeddings
        - Ranking Model
        """
        
        # Connect to Hopsworks
        project = hopsworks.connection().get_project()
        self.fs = project.get_feature_store()

        # 🔵🔵🔵🔵 Connect to Pinecone
        self.pc = Pinecone(api_key='83447319-a1d1-446b-bfbb-9fbce3071957')
        
        # Establish the 'articles' feature view
        self.articles_fv = self.fs.get_feature_view(
            name="articles", 
            version=1,
        )
        
        # Get list of feature names for articles
        self.articles_features = [feat.name for feat in self.articles_fv.schema]
        
        # Establish the 'customers' feature view
        self.customer_fv = self.fs.get_feature_view(
            name="customers", 
            version=1,
        )

        # 🔵🔵🔵🔵 Establish the 'candidate_embeddings' feature view
        self.pc_index = self.pc.Index("recsys-project")
        
        # # Retrieve the 'candidate_embeddings' feature view
        # self.candidate_index = self.fs.get_feature_view(
        #     name="candidate_embeddings", 
        #     version=1,
        # )

        # Retrieve ranking model
        mr = project.get_model_registry()
        model = mr.get_model(
            name="ranking_model", 
            version=1,
        )
        
        # Extract input schema from the model
        input_schema = model.model_schema["input_schema"]["columnar_schema"]
        
        # Get the names of features expected by the ranking model
        self.ranking_model_feature_names = [feat["name"] for feat in input_schema]

    def preprocess(self, inputs):
        # Extract the input instance
        inputs = inputs["instances"][0]
        
        # Extract customer_id from inputs
        customer_id = inputs["customer_id"]

        # 🔵🔵🔵🔵 Retrieve the 'candidate_embeddings' feature view
        neighbors = self.pc_index.query(
            vector=inputs["query_emb"],
            top_k=100,
            include_values=True)

        # Extract item_ids from returned object
        neighbors = [match['id'] for match in neighbors['matches']]
        
        # # ⚠️⚠️⚠️ Retrieve 100 candidate items from query vector
        # neighbors = self.candidate_index.find_neighbors(
        #     inputs["query_emb"], 
        #     k=100,
        # )

        # Extract item_ids from retured (id, embed) tuples
        # neighbors = [neighbor[0] for neighbor in neighbors]

        # BUILD ARTICLE DATA INPUTS
        # - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Filter Out previously purchased candidates
        # Enrich remaining with Article data
        
        # Get IDs of items already bought by the customer
        already_bought_items_ids = self.fs.sql(
            f"SELECT article_id from transactions_1 WHERE customer_id = '{customer_id}'"
        ).values.reshape(-1).tolist()

        # Filter candidate items to exclude those already bought by the customer
        item_id_list = [
            str(item_id) 
            for item_id 
            in neighbors 
            if str(item_id) 
            not in already_bought_items_ids
        ]
        # Create df from filtered list
        item_id_df = pd.DataFrame({"article_id" : item_id_list})
        
        # Retrieve Article data for candidate items
        articles_data = [
            self.articles_fv.get_feature_vector({"article_id": item_id}) 
            for item_id 
            in item_id_list
        ]

        articles_df = pd.DataFrame(
            data=articles_data, 
            columns=self.articles_features,
        )
        
        # Join candidate items with their features
        ranking_model_inputs = item_id_df.merge(
            articles_df, 
            on="article_id", 
            how="inner",
        )        

        # ADD CUSTOMER DATA
        # - - - - - - - - - - - - - - - - - - - - - 
        
        # Add customer features to ranking_model_inputs
        customer_features = self.customer_fv.get_feature_vector(
            {"customer_id": customer_id}, 
            return_type="pandas",
        )
        ranking_model_inputs["age"] = customer_features.age.values[0]   
        ranking_model_inputs["month_sin"] = inputs["month_sin"]
        ranking_model_inputs["month_cos"] = inputs["month_cos"]

        # Select only the features required by the ranking model
        ranking_model_inputs = ranking_model_inputs[self.ranking_model_feature_names]
                
        return { 
            "inputs" : [{"ranking_features": ranking_model_inputs.values.tolist(), "article_ids": item_id_list}]
        }
        
    # A simple postprocess method that just returns the model's output
    def postprocess(self, outputs):
        # If no postprocessing is needed, just return the outputs directly
        return outputs
