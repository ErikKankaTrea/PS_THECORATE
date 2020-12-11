import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import seaborn as sns
from annoy import AnnoyIndex
import re
import textwrap
import io
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from PIL import Image
from PIL.Image import Image as PilImage
import argparse
import json
import time


def get_data_path():
    with open("./settings.json") as f:
        dict_path = json.load(f)
    return dict_path['input_path']


def get_output_path():
    with open("./settings.json") as f:
        dict_path = json.load(f)
    return dict_path["output_path"]


def show_img(image):
    plt.imshow(image)



def imgs_to_pil(im_dir):
    from PIL import Image
    im = Image.open(im_dir)
    return im


def sub_structural_embedding(url_path):
    """Function that should be created
       to substract same features
    """
    #Apply transformation we defined in torch
    #Then pass it to the model (.cpu())
    #Substract embedding from model
    #Substract embedding of categories #THIS ONE PROBABLY IT SHOULD BE AN INFERENCE WITH A CLASSIFIER
    #Substract embedding of color
    return pic_embedding


class SimDistImg():
    def __init__(self, url_image, ds, top_sim):
        self.top_sim=6
        self.columns=5
        self.width=20
        self.height=8
        self.max_images=15 
        self.label_wrap_length=50
        self.label_font_size=8
        self.surname="Martinez"
        self.name="Erik"
        self.top_sim=top_sim
        self.url_image=url_image
        self.ds=ds


        self.f = len(self.ds['img_repr'][0])
        self.t = AnnoyIndex(self.f, metric='dot')
        ntree = 1000
        for i, vector in enumerate(self.ds['img_repr']):
            self.t.add_item(i, vector)
        _  = self.t.build(ntree)


    def get_similar_images_annoy(self):
        img_index=self.ds.loc[self.ds.uuid==self.url_image].index.tolist()
        base_img_id, base_vector  = self.ds.iloc[img_index[0], [0, 151]]
        similar_img_ids = self.t.get_nns_by_item(img_index[0], self.top_sim, include_distances=True)
        return base_img_id, similar_img_ids, self.ds.iloc[similar_img_ids[0]]


    def save_sim_images(self, images, data, features_df):
        
        self.height = max(self.height, int(len(images)/self.columns) * self.height)
        plt.figure(figsize=(self.width, self.height))

        for i, image in enumerate(images):
          id_img=data.uuid.iloc[i]
          if i==0:
            plt.figure(figsize=(self.width*2, self.height*2))
            plt.subplot(len(images) / self.columns + 1, self.columns, i + 1)  
          elif i==1:
            plt.figure(figsize=(self.width, self.height))
            plt.subplot(len(images) / self.columns + 1, self.columns, i)
          else:
            plt.subplot(len(images) / self.columns + 1, self.columns, i)
         
          plt.imshow(image)
          plt.axis('off')
          pic_info = features_df.loc[features_df.uuid==id_img, ["category", "provider", "material"]]
          footer = ' / \n'.join([pic_info["category"].values[0], pic_info["provider"].values[0], pic_info["material"].values[0]])
          plt.annotate(footer, (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points', va='top') 

        aux_date= time.strftime("%Y%m%d-%H%M%S")
        name_out = '{}_{}__{}.jpg'.format(self.surname, self.name, aux_date)
        plt.savefig(os.path.join(get_output_path(), name_out))




def url_to_pil(url:str):
    """Function to download picture, then should be processed on union embedding"""
    headers = { 'User-Agent': 'Mozilla/6.0'}
    data = urlopen(Request(url, headers=headers)).read()
    bts = io.BytesIO(data)
    im = Image.open(bts)
    im.filename = urlparse(url).path
    return im