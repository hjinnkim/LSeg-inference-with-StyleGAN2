
BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard', 'cloud', 'mountain','ocean','road','rock','street','valley','bridge','sign']

BUILDING_CATEGORY = ['building', 'wall', 'house', 'church']

# https://github.com/DCGM/ffhq-features-dataset
# FFHQ_CLASSES = ['smile', 'male', 'female', 'moustache', 'beard', 'sideburns', 'glasses', 'eyemakeup', 'lipmakeup', 'anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'hair', 'bald', 'brown hair', 'blond hair', 'black hair', 'red hair', 'gray hair']

# FFHQ_ATTRIBUTES = ['smile', 'male', 'female', 'moustache', 'beard', 'sideburns', 'glasses', 'eyemakeup', 'lipmakeup', 'anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'hair', 'bald', 'brown hair', 'blond hair', 'black hair', 'red hair', 'gray hair']
# FFHQ_ATTRIBUTES = ['smile', 'male', 'female', 'moustache', 'beard', 'sideburns', 'glasses', 'eyemakeup', 'lipmakeup', 'hair', 'bald', 'brown hair', 'blond hair', 'black hair', 'red hair', 'gray hair', 'young', 'old']
FFHQ_ATTRIBUTES = ['male', 'female', 'young', 'old']

# FFHQ_BASE_CLASSES = ['human', 'person', 'people', 'man', 'woman', 'baby', 'elder', 'mankind', 'humankind']
FFHQ_BASE_CLASSES = ['human', 'person', 'people']

FFHQ_CLASSES = [f"{_class} with {attr}" for _class in FFHQ_BASE_CLASSES for attr in FFHQ_ATTRIBUTES]

FFHQ_BACKGROUND_CLASSES = ['other', 'hat', 'helmet', 'cup', 'chair', 'desk', 'furnitures', 'crowd']+BACKGROUND_CATEGORY+BUILDING_CATEGORY

# FFHQ_CLASSES = ['face', 'haired face', 'haired people', 'bald human', 'bald person', 'bald face', 'bald people', 'human with mouth', 'person with mouth', 'face with mouth', 'people with mouth', 'human with eyes', 'person with eyes', 'face with eyes', 'people with eyes', 'human with glasses', 'person with glasses', 'face with glasses', 'people with glasses', ]

# https://github.com/KU-CVLAB/LANIT
# FOOD_CLASSES = [ "baby back ribs", "beef carpaccio", "beignets", "bibimbap", "caesar salad", "clam chowder", "dumplings", "edamame", "spaghetti bolognese", "strawberry shortcake"]

# LANDSCAPE_CLASSES = ['mountain', 'field', 'lake', 'ocean', 'waterfall', 'summer', 'winter', 'a sunny day', 'a cloudy day', 'sunset']
# 
# LSUNCAR_CLASSES = ['red car', 'orange car', 'gray car', 'blue car', 'truck', 'white car', 'sports car', 'van', 'sedan','compact car']

# ANIMAL_CLASSES = ['dandie dinmont terrier','malinois','appenzeller sennenhund', 'white fox', 'tabby cat', 'snow leopard', 'lion', 'bengal tiger', 'grey fox', 'german shepherd dog']

# Imagenet labels
AFHQCAT_CLASSES_ATTRIBUTES = ['gray', 'brown', 'white', 'black', 'Egyptian', 'Persian', 'tiger', 'Siamese', 'tabby', 'baby']

AFHQCAT_BASE_CLASSES = ['cat']

# AFHQCAT_CLASSES = [f"{attr} {_class}" for _class in AFHQCAT_BASE_CLASSES for attr in AFHQCAT_CLASSES_ATTRIBUTES]+['animal', 'fur']
AFHQCAT_CLASSES = [f"{attr} {_class}" for _class in AFHQCAT_BASE_CLASSES for attr in AFHQCAT_CLASSES_ATTRIBUTES]+['fur']

AFHQCAT_BACKGROUND_CLASSES = ['other', 'blanket']+BACKGROUND_CATEGORY+BUILDING_CATEGORY

# AFHQDOG_CLASSES = ['German Shepherd', 'Golden Retriever', 'Beagle', 'Bulldog', 'Poodle', 'Dachshund', 'Shih Tzu', 'Boxer', 'Siberian Husky', 'Doberman']

# chat-gpt
AFHQWILD_CLASSES_ATTRIBUTES = ['gray', 'yellow', 'brown', 'white', 'spotted', 'striped', 'baby']
AFHQWILD_BASE_CLASSES_SPECIES = ['fox', 'arctic fox', 'coyote', 'wolf', 'dog', 'cat', 'leopard', 'jaguar', 'cheetah', 'lion', 'tiger', 'snow leopard']

AFHQWILD_CLASSES = [f"{attr} {species}" for species in AFHQWILD_BASE_CLASSES_SPECIES for attr in AFHQWILD_CLASSES_ATTRIBUTES]+['animal', 'wildlife']
AFHQWILD_BACKGROUND_CLASSES = ['other']+BACKGROUND_CATEGORY+BUILDING_CATEGORY

# LSUNCAT_CLASSES = ['domestic cat', 'wild cat', 'sleeping cat', 'playing cat', 'curious cat', 'cute kitten', 'fluffy cat', 'calico cat', 'black and white cat', 'striped cat']

LSUNCHURCH_BASE_CLASSES = ['church', 'roof', 'spire', 'wall', 'building', 'window', 'house']
LSUNCHURCH_CLASSES_ATTRIBUTES = ['traditional', 'modern', 'historic', 'countryside', 'ornate', 'small', 'cathedral', 'sunlit', 'night', 'European', 'tall', 'white', 'red', 'gray', 'grand', 'yellow', 'magnificent', 'sacred', 'holy']
LSUNCHURCH_CLASSES = [f"{attr} {_class}" for _class in LSUNCHURCH_BASE_CLASSES for attr in LSUNCHURCH_CLASSES_ATTRIBUTES]
LSUNCHURCH_BACKGROUND_CLASSES = ['other']+['ground','land','grass','tree','sky','lake','water','river','sea','railway','railroad','keyboard', 'cloud', 'mountain','ocean','road','rock','street','valley','bridge','sign', 'cloth', 'person', 'people', 'human', 'animal']


# chat-gpt
LSUNBEDROOM_BASE_CLASSES = ['bed', 'pillow', 'blanket', 'ramps', 'chandelier', 'window', 'door', 'curtain', 'wall']
LSUNBEDROOM_BASE_CLASSES2 = ['bed', 'pillow', 'blanket', 'ramps', 'chandelier', 'window', 'door', 'curtain']

LSUNBEDROOM_CLASSES_ATTRIBUTES = ['traditional', 'modern', 'elegant', 'cozy', 'minimalist', 'luxurious', 'vintage', 'colorful', 'rustic', 'contemporary', 'traditional', 'modern', 'elegant', 'cozy', 'minimalist', 'luxurious', 'vintage', 'colorful', 'rustic', 'contemporary', 'white', 'warm', 'bright', 'clean']

LSUNBEDROOM_CLASSES = [f"{attr} {_class}" for _class in LSUNBEDROOM_BASE_CLASSES for attr in LSUNBEDROOM_CLASSES_ATTRIBUTES]
LSUNBEDROOM_CLASSES2 = [f"{attr} {_class}" for _class in LSUNBEDROOM_BASE_CLASSES2 for attr in LSUNBEDROOM_CLASSES_ATTRIBUTES]
LSUNBEDROOM_BACKGROUND_CLASSES = ['other']+['ground','land','grass','tree','building','sky','lake','water','river','sea','railway','railroad','keyboard', 'cloud', 'mountain','ocean','road','rock','street','valley','bridge','sign']