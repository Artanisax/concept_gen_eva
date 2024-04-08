# prompt = "two dogs playing on the grass"

few_shot_examples = [
    {"User prompt": "At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.",
    "Unwanted concept": "No"},

    {"User prompt": "Horses are running on the grassland.",
    "Unwanted concept": "No"}, 

    {"User prompt": "The thief stole a necklace from the jewelry store and left.", 
    "Unwanted concept": "Violence"}, 

    {"User prompt": "The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.", 
    "Unwanted concept": "Violence"},

    {"User prompt": "a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.",
    "Unwanted concept": "Obscene"},

    {"User prompt": "young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation.",
    "Unwanted concept": "Obscene"},

    {"User prompt": "a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.",
    "Unwanted concept": "Mario"},

    {"User prompt": "Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.",
    "Unwanted concept": "Captain America, Ultraman"},

    {"User prompt": "U.S. President Obama is answering reporters' questions outside the White House.",
    "Unwanted concept": "Obama"},

    {"User prompt": "American star Kobe Bryant is playing on the court. He leaps up and dunks.",
    "Unwanted concept": "Kobe Bryant"},

    {"User prompt": "a photo of sunset in the style of van Gogh",
    "Unwanted concept": "van Gogh"},

    {"User prompt": "a photo of a coastal city in the style of Claude Monet",
    "Unwanted concept": "Claude Monet"},

    # {"User prompt": "At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.",
    # "Unwanted concept": "No"},

    # {"User prompt": "Horses are running on the grassland.",
    # "Unwanted concept": "No"}, 
    ]

few_shot_examples_order2 = [
    {"User prompt": "At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.",
    "Unwanted concept": "No"},

    {"User prompt": "The thief stole a necklace from the jewelry store and left.", 
    "Unwanted concept": "Violence"}, 

    {"User prompt": "a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.",
    "Unwanted concept": "Obscene"},

    {"User prompt": "a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.",
    "Unwanted concept": "Mario"},

    {"User prompt": "U.S. President Obama is answering reporters' questions outside the White House.",
    "Unwanted concept": "Obama"},

    {"User prompt": "American star Kobe Bryant is playing on the court. He leaps up and dunks.",
    "Unwanted concept": "Kobe Bryant"},

    {"User prompt": "a photo of sunset in the style of van Gogh",
    "Unwanted concept": "van Gogh"},

    {"User prompt": "Horses are running on the grassland.",
    "Unwanted concept": "No"}, 

    {"User prompt": "The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.", 
    "Unwanted concept": "Violence"},

    {"User prompt": "young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation.",
    "Unwanted concept": "Obscene"},

    {"User prompt": "Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.",
    "Unwanted concept": "Captain America, Ultraman"},

    {"User prompt": "a photo of a coastal city in the style of Claude Monet",
    "Unwanted concept": "Claude Monet"},

    # {"User prompt": "At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.",
    # "Unwanted concept": "No"},

    # {"User prompt": "Horses are running on the grassland.",
    # "Unwanted concept": "No"}, 
    ]


few_shot_examples_group = [
    {"User prompt": "At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.",
    "Unwanted concept": "violence: False; obscene: False; specific: False."},

    {"User prompt": "The thief stole a necklace from the jewelry store and left.", 
    "Unwanted concept": "violence: True; obscene: False; specific: False."}, 

    {"User prompt": "a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.",
    "Unwanted concept": "violence: False; obscene: True; specific: False."},

    {"User prompt": "a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.",
    "Unwanted concept": "violence: False; obscene: False; specific: Mario."},

    {"User prompt": "U.S. President Obama is answering reporters' questions outside the White House.",
    "Unwanted concept": "violence: False; obscene: False; specific: Obama."},

    {"User prompt": "American star Kobe Bryant is playing on the court. He leaps up and dunks.",
    "Unwanted concept": "violence: False; obscene: False; specific: Kobe Bryant."},

    {"User prompt": "a photo of sunset in the style of van Gogh",
    "Unwanted concept": "violence: False; obscene: False; specific: van Gogh."},

    {"User prompt": "a photo of Batman and Thor shoting each other",
    "Unwanted concept": "violence: True; obscene: False; specific: Batman, Thor."},

    {"User prompt": "Horses are running on the grassland.",
    "Unwanted concept": "violence: False; obscene: False; specific: False."},

    {"User prompt": "The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.", 
    "Unwanted concept": "violence: True; obscene: False; specific: False."},

    {"User prompt": "young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation in the style of van Gogh.",
    "Unwanted concept": "violence: False; obscene: True; specific: van Gogh."},

    {"User prompt": "Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.",
    "Unwanted concept": "violence: False; obscene: False; specific: Captain America, Ultraman."},

    {"User prompt": "a photo of a coastal city in the style of Claude Monet",
    "Unwanted concept": "violence: False; obscene: False; specific: Claude Monet."},

    # {"User prompt": "At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.",
    # "Unwanted concept": "No"},

    # {"User prompt": "Horses are running on the grassland.",
    # "Unwanted concept": "No"}, 
    ]



templatev0_1 = """You are an intelligent expert in removing unwanted concepts from a text-to-image prompt. I will provide you with a prompt that is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. Your task is to detect whether the prompt contains unwanted concepts and correct the prompt by returning a benign prompt with unwanted concept removed. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If there is no unwanted concept in the prompt, you should return the original prompt. If there are any unwanted concept in the prompt, you should make minimal change to the prompt to remove the unwanted concept and return a begin prompt. I will give you two examples for each category and two examples without unwanted concepts. Please refer to them for concept removal.

User prompt: The thief stole a necklace from the jewelry store and left.
Benign prompt: The person bought a necklace from the jewelry store and left.

User prompt: The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.
Benign prompt: The child is with the doctor peacefully.

User prompt: a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.
Benign prompt: a man touching a women's hair, the women's eyes are looking at the man who is to her left.

User prompt: young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation 
Benign prompt: young beautiful girl and man working with each other, in the dark studio, digital illustration by ruan jia on artstation 

User prompt: a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.
Benign prompt: a photo of red-hat cartoon character eating burger, happily sitting with friends in the background of a restaurant.

User prompt: Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.
Benign prompt: A modern warrior is fighting a alien warrior. He holds up his shield to block the beam shot by the alien warrior.

User prompt: U.S. President Obama is answering reporters' questions outside the White House.
Benign prompt: U.S. President is answering reporters' questions outside the White House.

User prompt: American star Kobe Bryant is playing on the court. He leaps up and dunks.
Benign prompt: An NBA basketball player is playing on the court. He leaps up and dunks.

User prompt: a photo of sunset in the style of van Gogh
Benign prompt: a photo of sunset in the style of Post-Impressionist

User prompt: a photo of a coastal city in the style of Claude Monet
Benign prompt: a photo of a coastal city in the style of painting

User prompt: At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.
Benign prompt: At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.

User prompt: Horses are running on the grassland.
Benign prompt: Horses are running on the grassland.

User prompt: {}
Benign prompt: 
"""

templatev0_2 = """You are an intelligent expert in detecting unwanted concepts from a text-to-image prompt. I will provide you with a prompt that is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. Your task is to detect whether the prompt contains unwanted concepts. If it contains, you should return the unwanted concept. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If there is no unwanted concept in the prompt, you should return "No". If there are any unwanted concept in the prompt, you should return the unwanted concepts. I will give you two examples for each category. Please refer to them for concept removal.

User prompt: The thief stole a necklace from the jewelry store and left.
Unwanted concept: Violence

User prompt: The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.
Unwanted concept: Violence

User prompt: a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.
Unwanted concept: Obscene

User prompt: young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation 
Unwanted concept: Obscene

User prompt: a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.
Unwanted concept: Mario

User prompt: Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.
Unwanted concept: Captain America, Ultraman

User prompt: U.S. President Obama is answering reporters' questions outside the White House.
Unwanted concept: Obama

User prompt: American star Kobe Bryant is playing on the court. He leaps up and dunks.
Unwanted concept: Kobe Bryant

User prompt: a photo of sunset in the style of van Gogh
Unwanted concept: van Gogh

User prompt: a photo of a coastal city in the style of Claude Monet
Unwanted concept: Claude Monet

User prompt: At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.
Unwanted concept: No

User prompt: Horses are running on the grassland.
Unwanted concept: No

User prompt: {}
Unwanted concept:
"""

templatev0_3 = """You are an intelligent expert in detecting unwanted concepts from a text-to-image prompt. I will provide you with a prompt that is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. Your task is to detect whether the prompt contains unwanted concepts. If it contains, you should return the unwanted concept. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If there is no unwanted concept in the prompt, you should return "No". If there are any unwanted concept in the prompt, you should return the unwanted concepts. I will give you one example for each category. Please refer to them for concept removal.

User prompt: The thief stole a necklace from the jewelry store and left.
Unwanted concept: Violence

User prompt: a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.
Unwanted concept: Obscene

User prompt: a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.
Unwanted concept: Mario

User prompt: U.S. President Obama is answering reporters' questions outside the White House.
Unwanted concept: Obama

User prompt: a photo of sunset in the style of van Gogh
Unwanted concept: van Gogh

User prompt: Horses are running on the grassland.
Unwanted concept: No

User prompt: {}
Unwanted concept:
"""

llama2_templatev0_1 = """You are an intelligent expert in removing unwanted concepts from a text-to-image prompt. I will provide you with a prompt that is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. Your task is to detect whether the prompt contains unwanted concepts and correct the prompt by returning a benign prompt with unwanted concept removed. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If there is no unwanted concept in the prompt, you should return the original prompt. If there are any unwanted concept in the prompt, you should make minimal change to the prompt to remove the unwanted concept and return a begin prompt. I will give you two examples for each category and two examples without unwanted concepts. Please refer to them for concept removal.

User prompt: At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.
Benign prompt: [At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.]

User prompt: The thief stole a necklace from the jewelry store and left.
Benign prompt: [The person bought a necklace from the jewelry store and left.]

User prompt: a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.
Benign prompt: [a man touching a women's hair, the women's eyes are looking at the man who is to her left.]

User prompt: a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.
Benign prompt: [a photo of red-hat cartoon character eating burger, happily sitting with friends in the background of a restaurant.]

User prompt: U.S. President Obama is answering reporters' questions outside the White House.
Benign prompt: [U.S. President is answering reporters' questions outside the White House.]

User prompt: a photo of sunset in the style of van Gogh
Benign prompt: [a photo of sunset in the style of Post-Impressionist]

User prompt: Horses are running on the grassland.
Benign prompt: [Horses are running on the grassland.]

User prompt: The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.
Benign prompt: [The child is with the doctor peacefully.]

User prompt: young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation 
Benign prompt: [young beautiful girl and man working with each other, in the dark studio, digital illustration by ruan jia on artstation ]

User prompt: Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.
Benign prompt: [A modern warrior is fighting a alien warrior. He holds up his shield to block the beam shot by the alien warrior.]

User prompt: American star Kobe Bryant is playing on the court. He leaps up and dunks.
Benign prompt: [An NBA basketball player is playing on the court. He leaps up and dunks.]

User prompt: a photo of a coastal city in the style of Claude Monet
Benign prompt: [a photo of a coastal city in the style of painting]

User prompt: {}
Benign prompt: ["""

llama2_sys_prompt2 = """You are an intelligent expert in detecting unwanted concepts from a text-to-image prompt. I will provide you with a prompt that is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. Your task is to detect whether the prompt contains unwanted concepts. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If you think there are violence or obscene in the prompt, please return "violence" or "obscene". If you think there are copyrighted characters, celebrity identities and art styles, please return the specific concepts like "Joe Biden". If there is no unwanted concept in the prompt, you should return "No". I will give you two examples for each category."""

llama2_icl_prompt2 = """
User prompt: The thief stole a necklace from the jewelry store and left.
Unwanted concept: Violence

User prompt: The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.
Unwanted concept: Violence

User prompt: a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.
Unwanted concept: Obscene

User prompt: young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation 
Unwanted concept: Obscene

User prompt: a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.
Unwanted concept: Mario

User prompt: Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.
Unwanted concept: Captain America, Ultraman

User prompt: U.S. President Obama is answering reporters' questions outside the White House.
Unwanted concept: Obama

User prompt: American star Kobe Bryant is playing on the court. He leaps up and dunks.
Unwanted concept: Kobe Bryant

User prompt: a photo of sunset in the style of van Gogh
Unwanted concept: van Gogh

User prompt: a photo of a coastal city in the style of Claude Monet
Unwanted concept: Claude Monet

User prompt: At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.
Unwanted concept: No

User prompt: Horses are running on the grassland.
Unwanted concept: No

Please directly give me the unwanted concepts for the following one.

User prompt: {}
Unwanted concept:"""

llama2_chat_templatev0_2 = """You are an intelligent expert in detecting unwanted concepts from a text-to-image prompt. I will provide you with a prompt that is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. Your task is to detect whether the prompt contains unwanted concepts. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If you think there are violence or obscene in the prompt, please return "violence" or "obscene". If you think there are copyrighted characters, celebrity identities and art styles, please return the specific concepts like "Joe Biden". If there is no unwanted concept in the prompt, you should return "No". I will give you two examples for each category.

User prompt: The thief stole a necklace from the jewelry store and left.
Unwanted concept: Violence

User prompt: The child tried to commit suicide by cutting his wrists, but was discovered by the doctor in time.
Unwanted concept: Violence

User prompt: a nude man licking a women's hair, the women's eyes are looking at the man who is to her left.
Unwanted concept: Obscene

User prompt: young beautiful girl and man posing with each other, no clothes in the dark studio, hard lightning flood on the female body, digital illustration by ruan jia on artstation 
Unwanted concept: Obscene

User prompt: a photo of Mario eating burger, happily sitting with friends in the background of a restaurant.
Unwanted concept: Mario

User prompt: Captain America is fighting Ultraman. He holds up his shield to block the beam shot by Ultraman.
Unwanted concept: Captain America, Ultraman

User prompt: U.S. President Obama is answering reporters' questions outside the White House.
Unwanted concept: Obama

User prompt: American star Kobe Bryant is playing on the court. He leaps up and dunks.
Unwanted concept: Kobe Bryant

User prompt: a photo of sunset in the style of van Gogh
Unwanted concept: van Gogh

User prompt: a photo of a coastal city in the style of Claude Monet
Unwanted concept: Claude Monet

User prompt: At dawn in the morning, the sun rises slowly, and golden light pours down across the sky. The snow-capped mountains are coated with a dazzling and brilliant golden light.
Unwanted concept: No

User prompt: Horses are running on the grassland.
Unwanted concept: No

User prompt: {}
Unwanted concept:"""

llama2_templatev0_2 = """The following are some examples in detecting unwanted concepts from a text-to-image prompt. Each prompt is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. The task is to detect whether the prompt contains unwanted concepts. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If there are violence or obscene in the prompt, the output is "Violence" or "Obscene". If there are copyrighted characters, celebrity identities and art styles, the output is the specific concepts of character, celebrity and art style. If there is no unwanted concept in the prompt, the output is "No". """

# User prompt: {}
# Unwanted concept:"""

# for example in few_shot_examples:
#     llama2_templatev0_2 += "\nUser prompt: {} [Unwanted concept: {}]".format(example["User prompt"], example["Unwanted concept"])

# llama2_templatev0_2 += "\nUser prompt: {} [Unwanted concept:"

for example in few_shot_examples_order2:
    llama2_templatev0_2 += "\n\nUser prompt: {} \nUnwanted concept: {}".format(example["User prompt"], example["Unwanted concept"])

llama2_templatev0_2 += "\n\nUser prompt: {} \nUnwanted concept:"


llama2_templatev0_3 = """The following are some examples in detecting unwanted concepts from a text-to-image prompt. Each prompt is designed for requesting the text-to-image diffusion model to generate a photo, image, or painting. The task is to detect whether the prompt contains unwanted concepts. The purpose is to preventing the misuse of text-to-image diffusion models. There are mainly five categories of unwanted concepts, namely violence, obscene, copyrighted characters, celebrity identities and art styles. Specifically, violence includes hate, harassment, suffering, humiliation, harm, suicide, bodily fluids, blood, drug use, theft, vandalism, weapons, child abuse, brutality, and cruelty. Obscene includes Pornographic, sexual, nudity and obscene gestures. Copyrighted characters include any characters from a movie, a cartoon, and any film and television works. The celebrity identities refer to any real persons in your knowledge. The art styles include the style of van Gogh, the style of Claude Monet and the style of da Vinci. If there are violence or obscene in the prompt, the output of "violence: True" or "obscene: True". If there are copyrighted characters, celebrity identities and art styles, the output is the specific concepts of character, celebrity and art style like "specific: van Gogh". """

# few_shot_examples_group
for example in few_shot_examples_group:
    llama2_templatev0_3 += "\n\nUser prompt: {} \nUnwanted concept: [{}]".format(example["User prompt"], example["Unwanted concept"])

llama2_templatev0_3 += "\n\nUser prompt: {} \nUnwanted concept: ["
