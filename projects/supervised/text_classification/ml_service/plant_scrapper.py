# This script is used to scrape the plant species from the internet and save them to a file
# using beautifulsoup4 and requests

import requests
from bs4 import BeautifulSoup

# Top 25 Houseplants

""" Spider Plant: https://en.wikipedia.org/wiki/Chlorophytum_comosum
Snake Plant: https://en.wikipedia.org/wiki/Dracaena_trifasciata
ZZ Plant: https://en.wikipedia.org/wiki/Zamioculcas_zamiifolia
Peace Lily: https://en.wikipedia.org/wiki/Spathiphyllum
Monstera Deliciosa: https://en.wikipedia.org/wiki/Monstera_deliciosa
Pothos: https://en.wikipedia.org/wiki/Epipremnum_aureum
Aloe Vera: https://en.wikipedia.org/wiki/Aloe_vera
Rubber Plant: https://en.wikipedia.org/wiki/Ficus_elastica
Fiddle Leaf Fig: https://en.wikipedia.org/wiki/Ficus_lyrata
Prayer Plant: https://en.wikipedia.org/wiki/Maranta_leuconeura
Chinese Evergreen: https://en.wikipedia.org/wiki/Aglaonema
Philodendron: https://en.wikipedia.org/wiki/Philodendron
Cast Iron Plant: https://en.wikipedia.org/wiki/Aspidistra_elatior
Boston Fern: https://en.wikipedia.org/wiki/Nephrolepis_exaltata
Bird's Nest Fern: https://en.wikipedia.org/wiki/Asplenium_nidus
Air Plants (Tillandsia): https://en.wikipedia.org/wiki/Tillandsia
Succulents (general): https://en.wikipedia.org/wiki/Succulent_plant
Cactus (general): https://en.wikipedia.org/wiki/Cactus
Orchid (general): https://en.wikipedia.org/wiki/Orchid
African Violet: https://en.wikipedia.org/wiki/Saintpaulia
Begonia: https://en.wikipedia.org/wiki/Begonia
Dieffenbachia: https://en.wikipedia.org/wiki/Dieffenbachia
Dracaena Marginata: https://en.wikipedia.org/wiki/Dracaena_marginata
English Ivy: https://en.wikipedia.org/wiki/Hedera_helix
Money Tree: https://en.wikipedia.org/wiki/Pachira_aquatica
Top 25 Garden/Grocery Vegetables, Fruits, etc.

Tomato: https://en.wikipedia.org/wiki/Tomato
Potato: https://en.wikipedia.org/wiki/Potato
Onion: https://en.wikipedia.org/wiki/Onion
Garlic: https://en.wikipedia.org/wiki/Garlic
Carrot: https://en.wikipedia.org/wiki/Carrot
Lettuce: https://en.wikipedia.org/wiki/Lettuce
Cucumber: https://en.wikipedia.org/wiki/Cucumber
Bell Pepper: https://en.wikipedia.org/wiki/Bell_pepper
Broccoli: https://en.wikipedia.org/wiki/Broccoli
Cauliflower: https://en.wikipedia.org/wiki/Cauliflower
Spinach: https://en.wikipedia.org/wiki/Spinach
Apple: https://en.wikipedia.org/wiki/Apple
Banana: https://en.wikipedia.org/wiki/Banana
Orange: [invalid URL removed])
Strawberry: https://en.wikipedia.org/wiki/Strawberry
Blueberry: https://en.wikipedia.org/wiki/Blueberry
Raspberry: https://en.wikipedia.org/wiki/Raspberry
Avocado: https://en.wikipedia.org/wiki/Avocado
Lemon: https://en.wikipedia.org/wiki/Lemon
Lime: [invalid URL removed])
Basil: https://en.wikipedia.org/wiki/Basil
Rosemary: https://en.wikipedia.org/wiki/Rosemary
Thyme: https://en.wikipedia.org/wiki/Thyme
Mint: https://en.wikipedia.org/wiki/Mint
Parsley: https://en.wikipedia.org/wiki/Parsley """

plant_species_urls = {
  "Spider Plant": "https://en.wikipedia.org/wiki/Chlorophytum_comosum",
  "Snake Plant": "https://en.wikipedia.org/wiki/Dracaena_trifasciata",
  "ZZ Plant": "https://en.wikipedia.org/wiki/Zamioculcas_zamiifolia",
  "Peace Lily": "https://en.wikipedia.org/wiki/Spathiphyllum",
  "Monstera Deliciosa": "https://en.wikipedia.org/wiki/Monstera_deliciosa",
  "Pothos": "https://en.wikipedia.org/wiki/Epipremnum_aureum",
  "Aloe Vera": "https://en.wikipedia.org/wiki/Aloe_vera",
  "Rubber Plant": "https://en.wikipedia.org/wiki/Ficus_elastica",
  "Fiddle Leaf Fig": "https://en.wikipedia.org/wiki/Ficus_lyrata",
  "Prayer Plant": "https://en.wikipedia.org/wiki/Maranta_leuconeura",
  "Chinese Evergreen": "https://en.wikipedia.org/wiki/Aglaonema",
  "Philodendron": "https://en.wikipedia.org/wiki/Philodendron",
  "Cast Iron Plant": "https://en.wikipedia.org/wiki/Aspidistra_elatior",
  "Boston Fern": "https://en.wikipedia.org/wiki/Nephrolepis_exaltata",
  "Bird's Nest Fern": "https://en.wikipedia.org/wiki/Asplenium_nidus",
  "Air Plants (Tillandsia)": "https://en.wikipedia.org/wiki/Tillandsia",
  "Succulents (general)": "https://en.wikipedia.org/wiki/Succulent_plant",
  "Cactus (general)": "https://en.wikipedia.org/wiki/Cactus",
  "Orchid (general)": "https://en.wikipedia.org/wiki/Orchid",
  "African Violet": "https://en.wikipedia.org/wiki/Saintpaulia",
  "Begonia": "https://en.wikipedia.org/wiki/Begonia",
  "Dieffenbachia": "https://en.wikipedia.org/wiki/Dieffenbachia",
  "Dracaena Marginata": "https://en.wikipedia.org/wiki/Dracaena_marginata",
  "English Ivy": "https://en.wikipedia.org/wiki/Hedera_helix",
  "Money Tree": "https://en.wikipedia.org/wiki/Pachira_aquatica",
  "Tomato": "https://en.wikipedia.org/wiki/Tomato",
  "Potato": "https://en.wikipedia.org/wiki/Potato",
  "Onion": "https://en.wikipedia.org/wiki/Onion",
  "Garlic": "https://en.wikipedia.org/wiki/Garlic",
  "Carrot": "https://en.wikipedia.org/wiki/Carrot",
  "Lettuce": "https://en.wikipedia.org/wiki/Lettuce",
  "Cucumber": "https://en.wikipedia.org/wiki/Cucumber",
  "Bell Pepper": "https://en.wikipedia.org/wiki/Bell_pepper",
  "Broccoli": "https://en.wikipedia.org/wiki/Broccoli",
  "Cauliflower": "https://en.wikipedia.org/wiki/Cauliflower",
  "Spinach": "https://en.wikipedia.org/wiki/Spinach",
  "Apple": "https://en.wikipedia.org/wiki/Apple",
  "Banana": "https://en.wikipedia.org/wiki/Banana",
  "Orange": "https://en.wikipedia.org/wiki/Orange",
  "Strawberry": "https://en.wikipedia.org/wiki/Strawberry",
  "Blueberry": "https://en.wikipedia.org/wiki/Blueberry",
  "Raspberry": "https://en.wikipedia.org/wiki/Raspberry",
  "Avocado": "https://en.wikipedia.org/wiki/Avocado",
  "Lemon": "https://en.wikipedia.org/wiki/Lemon",
  "Lime": "https://en.wikipedia.org/wiki/Lime",
  "Basil": "https://en.wikipedia.org/wiki/Basil",
  "Rosemary": "https://en.wikipedia.org/wiki/Rosemary",
  "Thyme": "https://en.wikipedia.org/wiki/Thyme",
  "Mint": "https://en.wikipedia.org/wiki/Mint",
  "Parsley": "https://en.wikipedia.org/wiki/Parsley"
}

def scrape_plant_species():
    for plan_name, url in plant_species_urls.items():
        print(f"Scraping {plan_name}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.find_all('a')

if __name__ == "__main__":
    plant_species = scrape_plant_species()
    print(plant_species)
