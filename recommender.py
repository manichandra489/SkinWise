
import torch
import clip
import PIL.Image
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from itertools import product
from pydantic import BaseModel,field_validator, ConfigDict
from typing import Optional,List
from typing import Annotated
import pandas as pd
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
import chromedriver_autoinstaller
import re
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict
import joblib
import numpy as np
import PIL.Image # Changed from 'from PIL import Image'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import streamlit as st
# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
modeli, preprocess = clip.load("ViT-B/32", device=device)

"""Converting the Dataset Images into Vectors embeddings

Instantiated the Index in the PineCone Vector Database.
"""

pc = Pinecone(api_key="pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")  #pcsk_xEske_CL1K1Kxm8Zu2ncNFPaVW9TKroYJHYJn7KYb6Vtug66GUF5q8mFPLN9JbpWkGhgP

index_name = "skindisease-symptoms-gpt-4"

# Create new index if it doesn’t exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

"""Uploaded the Vector

Vector Search in RAG
"""

#if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#image_path = os.path.join(image_folder, filename)

def RAG(image,index_name,api):
  image = preprocess(PIL.Image.open(image)).unsqueeze(0).to(device)
  pc = Pinecone(api_key=api)
  index = pc.Index(index_name)
  with torch.no_grad():
      image_features = modeli.encode_image(image)

  image_features /= image_features.norm(dim=-1, keepdim=True)
  query_vector = image_features.cpu().numpy().flatten().tolist()
  results = index.query(
      vector=query_vector,
      top_k=2,
      include_metadata=True
  )
  results = results["matches"]
  return results
"""State class to hold the information throughout the workflow"""
class State(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)
  image: Optional[str] = None
  eligible:bool=False
  age: Optional[str] = None
  gender: Optional[str] = None
  bauman_type: Annotated[str, operator.add]
  skin_disease: Optional[str] = None
  toxic: Optional[str] = None
  weblinks: Annotated[List[str], operator.add] = []
  products: pd.DataFrame = pd.DataFrame()

  @field_validator("products")
  @classmethod
  def validate_df(cls, products: pd.DataFrame):
      expected_columns = {
          "product id":int,
          "Title":str,
          "Subtitle":str,
          "Price":str,
          "Rating":str,
          "Link":str,
          "Img_url":str
          }

      return products

"""LLM usage for the product recommendation"""

# Initialize client with your API key
llm = ChatOpenAI( # Renamed instance for clarity
      api_key=st.secrets["OpenAI_key"],
      model="gpt-4o-mini",  # You can use "gpt-4o", "gpt-5", etc.
      temperature=0.0)

def websearchWS(state: State):
    urls = []

    # Map specific skin diseases to URLs
    disease_urls = {
        "acne": "https://www.lorealparisusa.com/skin-care/dark-spots",
        "redness": "https://www.lorealparisusa.com/skin-care/dry-skin",
        "bags": "https://www.lorealparisusa.com/skin-care/sagging-skin",
    }

    # Map Bauman types (or parts of them) to multiple related URLs
    bauman_type_urls = {
        "anti-aging": [
            "https://www.lorealparisusa.com/skin-care/anti-aging",
            "https://www.lorealparisusa.com/skin-care/fine-lines-wrinkles",
        ],
        "dark circles": [
            "https://www.lorealparisusa.com/skin-care/dark-circles-under-eyes",
        ],
        "dry": [
            "https://www.lorealparisusa.com/skin-care/dry-skin",
            "https://www.lorealparisusa.com/skin-care/fragrance-free",
        ],
        "oily": [
            "https://www.lorealparisusa.com/skin-care/oily-skin",
        ],
    }

    # 1) Add disease-based URL (exact match or normalized)
    if state.skin_disease:
        key = state.skin_disease.strip().lower()
        if key in disease_urls:
            urls.append(disease_urls[key])

    # 2) Add Bauman-type-based URLs using partial / similar matches
    if state.bauman_type:
        bt = state.bauman_type.strip().lower()
        for pattern, pattern_urls in bauman_type_urls.items():
            if pattern in bt:      # "similar" / substring match
                urls.extend(pattern_urls)

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)

    return {"weblinks": unique_urls}

"""Web Scraping of the L'oreal Paris USA website to get the product details"""

def scrape_products(state:State):
  id1=[]
  title1=[]
  subtitle1=[]
  price1=[]
  rating1=[]
  link1=[]
  image_url1=[]
  # Use chromedriver_autoinstaller to get a compatible chromedriver
  chromedriver_autoinstaller.install()

  # Set up Chrome options
  options = ChromeOptions()
  options.add_argument("--headless=new")
  options.add_argument("--no-sandbox")
  options.add_argument("--disable-dev-shm-usage")
  options.add_argument("--window-size=1920,1080") # Add window size for headless
  options.add_argument("--disable-gpu") # Added for stability in headless environments
  options.add_argument("--disable-extensions") # Added for stability in headless environments
  options.add_argument("--remote-debugging-port=9222") # Added for stability

  service = Service()
  driver = webdriver.Chrome(service=service, options=options)

  try:
    for url in state.weblinks:
      try:
          driver.get(url)
          WebDriverWait(driver, 30).until( # Increased timeout to 30 seconds
              EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".oap-card__front"))
          )
          driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
          time.sleep(2)  # let lazy-loaded images load in
          scroll_height = driver.execute_script("return document.body.scrollHeight")
          for i in range(0, scroll_height, 400):
              driver.execute_script(f"window.scrollTo(0, {i});")
              time.sleep(0.5)

          html = driver.page_source
          soup = BeautifulSoup(html, "html.parser")
          products = soup.find_all("div", class_="oap-card__front")
          for idx, prod in enumerate(products, 1):
              img_tag = prod.select_one(".oap-card__thumbnail img")
              image_url = None
              if img_tag:
                  if img_tag.get('src') and 'png' in img_tag['src']:
                      image_url = img_tag['src']
                  elif img_tag.get('data-src'):
                      image_url = img_tag['data-src']
                  elif img_tag.get('srcset'):
                      srcset_imgs = [s.strip().split(' ')[0] for s in img_tag['srcset'].split(',')]
                      image_url = srcset_imgs[-1] if srcset_imgs else None
              if image_url and image_url.startswith('/'):
                  image_url = "https://www.lorealparisusa.com" + image_url


              subtitle = prod.select_one(".oap-card__subtitle")
              title = prod.select_one(".oap-card__title")
              link_parent = prod.select_one(".oap-card__link")
              link = "https://www.lorealparisusa.com" + link_parent['href'] if link_parent and 'href' in link_parent.attrs else None
              price = prod.select_one(".oap-card__price p")
              rating = prod.select_one(".oap-rating__average")

              id1.append(idx)
              title1.append(title.get_text(strip=True) if title else 'N/A')
              subtitle1.append(subtitle.get_text(strip=True) if subtitle else 'N/A')
              price1.append(price.get_text(strip=True) if price else 'N/A')
              rating1.append(rating.get_text(strip=True) if rating else 'N/A')
              link1.append(link if link else 'N/A')
              image_url1.append(image_url if image_url else 'N/A')

              # Neatly print product details
      except Exception as e:
          print(f"Error scraping URL {url}: {e}")
          continue # Continue to the next URL if there's an error on one

    produc = {"product id":id1,"Title":title1,"Subtitle":subtitle1,"Price":price1,"Rating":rating1,"Link":link1,"Img_url":image_url1}
    df = pd.DataFrame(produc)
    df_sorted = df.sort_values(by='Rating', ascending=False).head(3)
    return {'products': pd.DataFrame(df_sorted)}
  finally:
    driver.quit()

"""Extraction of the products from the generated text"""

def extract(state:State):
  matches = re.findall(r"\*\*(.*?)\*\*", state.des)
  return({'meds':matches})
  global meds
  meds=matches


"""The testing of compatability between products and Anti Toxic nature of products."""

def examiner(state:State):
  #for i in len(ingrediants)-1:
   # j = i+1
  messages=[
          SystemMessage(content="Act as a skin care specialist aware of harmful ingredients in skincare products that are harmfull to skin."),
          HumanMessage(content=f"Please state the anti toxic products to each other {state.products.Title} {state.products.Subtitle}. State if the products can be used together. If not state which ones should not be used together and only from the given products list only.")
      ]

  chat_response = llm.invoke(messages) # Corrected call and assigned to a new variable
  return {'toxic': chat_response.content} # Return the content

"""RAG for any skin diseases check before suggesting a product"""

skinapi="pcsk_xEske_CL1K1Kxm8Zu2ncNFPaVW9TKroYJHYJn7KYb6Vtug66GUF5q8mFPLN9JbpWkGhgP"

def skin_disease(state: State):
    result = RAG(
        image=state.image,
        index_name="clip-skd",
        api=skinapi
    )

    disease = result[0]["metadata"]["Disease"]

    if disease == "Normal":
        return "normal"
    else:
        return "abnormal"

#skin_disease()

"""Node for Elgibility check"""

def com(state: State):
    return {"Eligible": True}

"""RAG for any reducable skin diseases to recommend a product for it"""

def skin_disease_reduce(state:State):
  if RAG(state.image,"skindisease-symptoms-gpt-4","pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")[0]['score']<0.55:
    return {"skin_disease":''}
  return {"skin_disease":RAG(state.image,"skindisease-symptoms-gpt-4","pcsk_734a9H_98bCfspdWWc4XgFoZBPsYB2CNw498LBK53KNDyF1WjVGsBcN8rsTzSYVWCc1GkQ")[0]['metadata']['Disease']}

"""Testing the Reducable skin disease function.

RAG for oily skin to find the Baumann skin type and to recommend a product for it
"""

def oily(state:State):
  return {"bauman_type":f"{RAG(state.image,'clip-image-index',skinapi)[0]['metadata']['filename'].split('_')[0]} "}

"""RAG for oily skin to find the Baumann skin type and to recommend a product for it"""

def sense(state:State):
  if RAG(state.image,"clip-sens",skinapi)[0]['metadata']['filename'].split("_")[0]=='normal':
    return {"bauman_type":"resistant "}
  else:
    return {"bauman_type":"sensitive "}

"""RAG for oily skin to find the Baumann skin type and to recommend a product for it"""

def pig(state:State):
  if RAG(state.image,"clip-pig",skinapi)[0]['metadata']['filename'].split("_")[1]=='png':
    return {"bauman_type":'pigmentation '}
  else:
    return {"bauman_type":'non-pigmentation '}

"""RAG for oily skin to find the Baumann skin type and to recommend a product for it"""

def wri(state:State):
  if RAG(state.image,"clip-wri",skinapi)[0]['metadata']['filename'][0]=='w':
    return {"bauman_type":'wrinkle'}
  else:
    return {"bauman_type":'tight'}

"""Gender dictionary 0 is Male and 1 is Female. The classification of age in between Child, Adult, Man, Old"""

gender_dict = {0:"Male",1:"Female"}
def category(age):
  if age>=0 and age<18:
    age="Child"
  elif age>=18 and age<50:
    age="Adult"
  elif age>=50:
    age="Old"
  return age

"""Finding the age, gender of the skin from  the image using CNN neural network"""

def age(state:State):
  # Load the image as grayscale and resize
  model = load_model("aagp.keras")
  img = load_img(state.image, color_mode="grayscale", target_size=(128, 128))
  img = np.array(img)

  # Reshape to add batch dimension and channel dimension for Keras model input
  # Expected shape is (batch_size, height, width, channels)
  img = img.reshape(1, 128, 128, 1)

  pred = model.predict(img)
  pred_gender = gender_dict[round(pred[0][0][0])]
  pred_age = round(pred[1][0][0]/100)
  return {'gender':pred_gender,'age':category(pred_age)}

"""Creation of Nodes from the functions, Edges and Workflow for between different Agents
"""

builder = StateGraph(State)
builder.add_node("skin disease reduce", skin_disease_reduce)
builder.add_node("com", com)
builder.add_node("oily", oily)
builder.add_node("sense", sense)
builder.add_node("pig", pig)
builder.add_node("wri", wri)
builder.add_node("age", age)
builder.add_node("weblinks", websearchWS)
builder.add_node("webprod",scrape_products)
builder.add_node("examinerv2", examiner)

builder.add_conditional_edges(
    START,
    skin_disease,
    {
        "normal": "com",
        "abnormal": END
    }
)
builder.add_edge("com", "skin disease reduce")
builder.add_edge("com", "oily")
builder.add_edge("com", "sense")
builder.add_edge("com", "pig")
builder.add_edge("com", "wri")
builder.add_edge("com", "age")
builder.add_edge("skin disease reduce", "weblinks")
builder.add_edge("oily", "weblinks")
builder.add_edge("sense", "weblinks")
builder.add_edge("pig", "weblinks")
builder.add_edge("wri", "weblinks")
builder.add_edge("age", "weblinks")
builder.add_edge("weblinks", "webprod")
builder.add_edge("webprod", "examinerv2")
builder.add_edge("examinerv2", END)

graph = builder.compile()

"""Usage of the Langgraph workflow"""

class ImageState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Optional[Any] = None

def run(img_state: ImageState):
    messages = graph.invoke({"image": img_state.image,"products": pd.DataFrame()})
    print(messages)
    return messages
    
