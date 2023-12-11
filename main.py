from model.py import seq2seqModel
import nltk
nltk.download('punkt')

is_prod = True # production mode or not

if is_prod:
    model = seq2seqModel.load('pretrained_moodle.pt')

    to_test = ['I am a student.',
               'I have a red car.',  # inversion captured
               'I love playing video games.',
                'This river is full of fish.', # plein vs pleine (accord)
                'The fridge is full of food.',
               'The cat fell asleep on the mat.',
               'my brother likes pizza.', # pizza is translated to 'la pizza'
               'I did not mean to hurt you', # translation of mean in context
               'She is so mean',
               'Help me pick out a tie to go with this suit!', # right translation
               "I can't help but smoking weed", # this one and below: hallucination
               'The kids were playing hide and seek',
               'The cat fell asleep in front of the fireplace']

    for elt in to_test:
        print('= = = = = \n','%s -> %s' % (elt, model.predict(elt)))
